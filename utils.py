
import cv2
import pytesseract
import re
import json
import os
from datetime import datetime
import spacy
import streamlit as st
from spacy.training.example import Example
import sqlite3

DB_FILE = "corrections.db"
NER_MODEL_PATH = "ner_model/default/model-best"

@st.cache_resource(show_spinner="🔁 Loading NER model...")
def get_nlp(model_path=NER_MODEL_PATH):
    try:
        return spacy.load(model_path)
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
        kv_match = re.match(r"(.+?)\s*[:\-]\s*(.+)", line)
        if kv_match:
            key = kv_match.group(1).strip().title()
            value = kv_match.group(2).strip()
            info[key] = value
            continue

        dob_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', line)
        if dob_match:
            info['DOB'] = dob_match.group(1)

        id_match = re.search(r'\d{4}[\s-]?\d{4}[\s-]?\d{4}', line)
        if id_match:
            raw_id = id_match.group(0).replace(" ", "").replace("-", "")
            formatted_id = ' '.join([raw_id[i:i+4] for i in range(0, 12, 4)])
            info['ID'] = formatted_id

        if line.replace(" ", "").isalpha() and 3 <= len(line.split()) <= 3:
            if 'Name' not in info:
                info['Name'] = line.title()

    for k, v in info.items():
        st.write(f"Detected: {k} → {v}")

    return info

def parse_with_spacy(text, model_path=NER_MODEL_PATH):
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model not found at {model_path}")
        return {}

    try:
        nlp = spacy.load(model_path)
    except:
        st.warning("⚠️ Failed to load NER model.")
        return {}

    result = {}
    if not nlp:
        return result

    doc = nlp(text)
    if not doc.ents:
        st.warning("⚠️ NER model found no entities.")
        return result

    for ent in doc.ents:
        label = ent.label_.title().replace("_", " ")
        value = ent.text.strip()
        result[label] = value
        st.write(f"Detected: {label} → {value}")

    return result

def save_correction(ocr_text, corrected_data, doc_type, predicted_data=None):
    timestamp = datetime.now().isoformat()
    reward = compute_reward(predicted_data or {}, corrected_data)

    model_path = f"ner_model/{doc_type}/model-best"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ocr_text TEXT,
            predicted_fields TEXT,
            corrected_fields TEXT,
            reward REAL,
            doc_type TEXT
        )
        """)
        cursor.execute("""
        INSERT INTO corrections (timestamp, ocr_text, predicted_fields, corrected_fields, reward, doc_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            ocr_text,
            json.dumps(predicted_data or {}),
            json.dumps(corrected_data),
            reward,
            doc_type
        ))
        conn.commit()
        conn.close()
        st.success("✅ Correction saved to SQLite!")
        print(f"[save_correction] Saved correction for doc_type={doc_type}, reward={reward:.2f}")
    except Exception as e:
        st.error(f"[ERROR] Failed to save correction to DB: {e}")

def compute_reward(predicted, corrected):
    reward = 0
    total = len(corrected)
    normalized_predicted = {k.strip().lower(): v.strip().lower() for k, v in predicted.items()}
    for key, value in corrected.items():
        norm_key = key.strip().lower()
        norm_val = value.strip().lower()
        pred_val = normalized_predicted.get(norm_key)
        if pred_val and pred_val == norm_val:
            reward += 1
    return reward / total if total > 0 else 0

def update_model_on_correction(nlp, ocr_text, corrected):
    doc = nlp.make_doc(ocr_text)
    entities = []
    for label, value in corrected.items():
        match = re.search(re.escape(value), ocr_text, re.IGNORECASE)
        if match:
            start, end = match.span()
            entities.append((start, end, label.upper()))

    example = Example.from_dict(doc, {"entities": entities})
    optimizer = nlp.resume_training()
    losses = {}
    nlp.update([example], sgd=optimizer, losses=losses)
    return losses
