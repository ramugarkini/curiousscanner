import sqlite3
import json
import re
import spacy
import os
import subprocess
from spacy.tokens import DocBin
import sys
import io

# Ensure UTF-8 output for logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DB_FILE = "corrections.db"
NER_MODEL_BASE = "ner_model"
TRAINING_FILE = "training_data.spacy"

def fetch_logs(latest_only=False, doc_type=None):
    if not os.path.exists(DB_FILE):
        print(f"[warn] Database file '{DB_FILE}' not found.")
        return []

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT ocr_text, corrected_fields FROM corrections"
    if doc_type:
        query += " WHERE doc_type = ?"
        args = (doc_type,)
    else:
        args = ()

    if latest_only:
        query += " ORDER BY id DESC LIMIT 1"

    rows = cursor.execute(query, args).fetchall()
    conn.close()

    logs = []
    for text, corrected_json in rows:
        try:
            corrected = json.loads(corrected_json)
            logs.append({
                "ocr_text": text,
                "corrected_fields": corrected
            })
        except:
            continue

    return logs

def get_dynamic_label_map(logs):
    label_map = {}
    for entry in logs:
        for field in entry.get("corrected_fields", {}):
            if field not in label_map:
                label_map[field] = field.upper().replace(" ", "_")
    return label_map

def generate_training_data(logs, output_file):
    if not logs:
        print("[warn] No logs found for training.")
        return False

    nlp = spacy.blank("en")
    db = DocBin()
    label_map = get_dynamic_label_map(logs)

    for entry in logs:
        text = entry["ocr_text"]
        corrected = entry["corrected_fields"]
        entities = []

        for label, value in corrected.items():
            label_upper = label_map[label]
            match = re.search(re.escape(value), text, re.IGNORECASE)
            if match:
                start, end = match.span()
                entities.append((start, end, label_upper))
            else:
                print(f"[MISS] {label} '{value}' not found in OCR text.")

        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract") or \
                   doc.char_span(start, end, label=label, alignment_mode="expand")
            if span:
                spans.append(span)

        doc.ents = spans
        db.add(doc)

    db.to_disk(output_file)
    print(f"[ok] Saved training data: {output_file}")
    return True

if __name__ == "__main__":
    latest_only = "--latest-only" in sys.argv

    doc_type = None
    if "--doc-type" in sys.argv:
        idx = sys.argv.index("--doc-type") + 1
        if idx < len(sys.argv):
            doc_type = sys.argv[idx]

    print(f"[info] Training model for doc_type: {doc_type or 'default'}")

    logs = fetch_logs(latest_only=latest_only, doc_type=doc_type)
    print(f"[info] Found {len(logs)} correction(s) for training.")

    MODEL_OUTPUT_DIR = os.path.join(NER_MODEL_BASE, doc_type or "default")

    if generate_training_data(logs, TRAINING_FILE):
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        print(f"[info] Training spaCy model...")
        result = subprocess.run([
            "python", "-m", "spacy", "train", "config.cfg",
            "--output", MODEL_OUTPUT_DIR,
            "--paths.train", TRAINING_FILE,
            "--paths.dev", TRAINING_FILE
        ])
        if result.returncode == 0:
            print(f"[ok] Model saved to: {MODEL_OUTPUT_DIR}")
        else:
            print("[fail] Training failed.")
