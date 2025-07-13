import json
import re
import spacy
import os
import sys
from spacy.tokens import DocBin
from spacy.cli.train import train

INPUT_FILE = "corrections_log.json"
NER_MODEL_BASE = "ner_model"

def get_dynamic_label_map(logs):
    label_map = {}
    for entry in logs:
        for field in entry.get("corrected_fields", {}):
            if field not in label_map:
                label_map[field] = field.upper().replace(" ", "_")
    return label_map

def generate_training_data_for_type(doc_type):
    nlp = spacy.blank("en")
    db = DocBin()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    logs = [entry for entry in logs if entry.get("doc_type") == doc_type]

    if not logs:
        print(f"[warn] No logs found for type: {doc_type}")
        return

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
    doc_types = set()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)
        for entry in logs:
            if "doc_type" in entry:
                doc_types.add(entry["doc_type"])

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
    nlp = spacy.blank("en")
    db = DocBin()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    if not logs:
        print("[warn] No entries found.")
        return None, None

    latest = logs[-1]
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

