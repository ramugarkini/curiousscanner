import sqlite3
import json
import re
import spacy
import os
import sys
from spacy.tokens import DocBin
from spacy.cli.train import train

DB_FILE = "corrections.db"
NER_MODEL_BASE = "ner_model"

def fetch_logs(doc_type=None, latest_only=False):
    if not os.path.exists(DB_FILE):
        print(f"[warn] Database file '{DB_FILE}' not found.")
        return []

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    query = "SELECT doc_type, ocr_text, corrected_fields FROM corrections"
    params = ()
    if doc_type:
        query += " WHERE doc_type = ?"
        params = (doc_type,)

    if latest_only:
        query += " ORDER BY id DESC LIMIT 1"

    rows = cursor.execute(query, params).fetchall()
    conn.close()

    logs = []
    for dt, text, corrected_json in rows:
        try:
            corrected = json.loads(corrected_json)
            logs.append({
                "doc_type": dt,
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

def generate_training_data_for_type(doc_type):
    logs = fetch_logs(doc_type=doc_type)
    if not logs:
        print(f"[warn] No logs found for type: {doc_type}")
        return

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

    out_file = f"training_{doc_type}.spacy"
    db.to_disk(out_file)
    print(f"[ok] Saved training data: {out_file}")

def auto_train_models():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT doc_type FROM corrections")
    doc_types = [row[0] for row in cursor.fetchall()]
    conn.close()

    for doc_type in doc_types:
        print(f"\n🔁 Generating training data for '{doc_type}'...")
        generate_training_data_for_type(doc_type)

        output_dir = os.path.join(NER_MODEL_BASE, doc_type)
        os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 Training spaCy model for '{doc_type}'...")
        result = os.system(
            f"python -m spacy train config.cfg --output {output_dir} --paths.train training_{doc_type}.spacy --paths.dev training_{doc_type}.spacy"
        )

        if result == 0:
            print(f"[ok] Model for '{doc_type}' saved to: {output_dir}")
        else:
            print(f"[fail] Training failed for: {doc_type}")

def generate_training_data_latest():
    logs = fetch_logs(latest_only=True)
    if not logs:
        print("[warn] No entries found.")
        return None, None

    latest = logs[0]
    doc_type = latest.get("doc_type", "unknown")
    label_map = get_dynamic_label_map([latest])

    text = latest["ocr_text"]
    corrected = latest["corrected_fields"]
    entities = []

    for label, value in corrected.items():
        label_upper = label_map[label]
        match = re.search(re.escape(value), text, re.IGNORECASE)
        if match:
            start, end = match.span()
            entities.append((start, end, label_upper))

    nlp = spacy.blank("en")
    db = DocBin()
    doc = nlp.make_doc(text)
    spans = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label, alignment_mode="contract") or \
               doc.char_span(start, end, label=label, alignment_mode="expand")
        if span:
            spans.append(span)

    doc.ents = spans
    db.add(doc)

    out_file = f"training_{doc_type}_latest.spacy"
    db.to_disk(out_file)
    print(f"[ok] Saved latest-only training file: {out_file}")
    return out_file, doc_type

if __name__ == "__main__":
    out_file, doc_type = generate_training_data_latest()
    if out_file and doc_type:
        output_dir = os.path.join(NER_MODEL_BASE, doc_type)
        os.makedirs(output_dir, exist_ok=True)
        print(f"🚀 Training spaCy model for latest '{doc_type}' correction...")
        result = os.system(
            f"python -m spacy train config.cfg --output {output_dir} --paths.train {out_file} --paths.dev {out_file}"
        )
        if result == 0:
            print(f"[ok] Latest model trained and saved to {output_dir}")
