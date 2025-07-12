# utils.py
import cv2
import pytesseract
import re
import json
import os
from datetime import datetime

CORRECTION_LOG = "corrections_log.json"

def preprocess_image(image, use_adaptive=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if use_adaptive:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    else:
        return gray

def extract_text(image):
    return pytesseract.image_to_string(image)

def parse_fields(text):
    info = {}
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    for line in lines:
        if len(line.split()) >= 2:
            info['Name'] = line.title()
            break

    dob_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', text)
    if dob_match:
        info['DOB'] = dob_match.group(1)

    id_match = re.search(r'(\d{12})|(\d{4}\s\d{4}\s\d{4})', text.replace(" ", ""))
    if id_match:
        raw_id = id_match.group(0)
        formatted_id = ' '.join([raw_id[i:i+4] for i in range(0, 12, 4)])
        info['ID Number'] = formatted_id

    return info

def save_correction(ocr_text, corrected_data):
    if not os.path.exists(CORRECTION_LOG):
        with open(CORRECTION_LOG, "w") as f:
            json.dump([], f)

    try:
        with open(CORRECTION_LOG, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []  # fallback if file is empty or corrupted

            data.append({
                "timestamp": datetime.now().isoformat(),
                "ocr_text": ocr_text,
                "corrected_fields": corrected_data
            })

            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save correction: {e}")

