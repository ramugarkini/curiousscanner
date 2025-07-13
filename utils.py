import cv2
import pytesseract
import re
import json
import os
from datetime import datetime
import spacy
import streamlit as st

CORRECTION_LOG = "corrections_log.json"
NER_MODEL_PATH = "ner_model/model-best"

@st.cache_resource
def get_nlp():
    try:
        return spacy.load(NER_MODEL_PATH)
    except Exception as e:
        st.warning(f"⚠️ Failed to load NER model: {e}")
        return None

def preprocess_image(image, use_adaptive=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if use_adaptive:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    return gray

def extract_text(image):
    return pytesseract.image_to_string(image)

def parse_fields(text):
    info = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    for line in lines:
        if "ramu" in line.lower():
            info['Name'] = line.title()

        if "admin" in line.lower() and ':' in line:
            info['Admin No'] = line.split(":")[-1].strip()

        if "branch" in line.lower():
            info['Branch'] = line.split(":")[-1].strip()
        elif line.strip().startswith(": CSE"):
            info['Branch'] = "CSE"

        if "contact" in line.lower() or "etNo" in line:
            digits = re.findall(r'\d{10}', line)
            if digits:
                info['Contact No'] = digits[0]

        dob_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', line)
        if dob_match:
            info['DOB'] = dob_match.group(1)

        id_match = re.search(r'\d{4}[\s-]?\d{4}[\s-]?\d{4}', line)
        if id_match:
            raw_id = id_match.group(0).replace(" ", "").replace("-", "")
            formatted_id = ' '.join([raw_id[i:i+4] for i in range(0, 12, 4)])
            info['ID'] = formatted_id

    return info

def parse_with_spacy(text):
    nlp = get_nlp()
    result = {}
    if not nlp:
        return result

    doc = nlp(text)
    for ent in doc.ents:
        label = ent.label_.title().replace("_", " ")
        value = ent.text.strip()
        result[label] = value
        st.write(f"Detected: {label} → {value}")

    return result

def save_correction(ocr_text, corrected_data):
    if not os.path.exists(CORRECTION_LOG):
        with open(CORRECTION_LOG, "w") as f:
            json.dump([], f)

    try:
        with open(CORRECTION_LOG, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

            data.append({
                "timestamp": datetime.now().isoformat(),
                "ocr_text": ocr_text,
                "corrected_fields": corrected_data
            })

            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"[ERROR] Failed to save correction: {e}")
