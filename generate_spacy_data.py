import json
import re
import spacy
from spacy.tokens import DocBin

INPUT_FILE = "corrections_log.json"
OUTPUT_FILE = "training_data.spacy"

def get_dynamic_label_map(logs):
    label_map = {}
    for entry in logs:
        for field in entry.get("corrected_fields", {}):
            if field not in label_map:
                label_map[field] = field.upper().replace(" ", "_")
    return label_map

def generate_training_data():
    nlp = spacy.blank("en")
    db = DocBin()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    label_map = get_dynamic_label_map(logs)

    for entry in logs:
        text = entry["ocr_text"]
        corrected = entry["corrected_fields"]
        entities = []

        print(f"\n[ENTRY] OCR Text:\n{text.strip()}")

        for label, value in corrected.items():
            label_upper = label_map[label]

            # Match ID number formats (e.g., Aadhaar-style 12-digit)
            # if label.upper() == "ID":
            #     id_clean = re.sub(r"\D", "", value)
            #     found = False
            #     for match in re.finditer(r'\d{4}[\s\-]?\d{4}[\s\-]?\d{4}', text):
            #         candidate = match.group(0)
            #         candidate_clean = re.sub(r"\D", "", candidate)
            #         if candidate_clean == id_clean:
            #             start, end = match.span()
            #             entities.append((start, end, label_upper))
            #             print(f"[MATCH] ID → {candidate} ({start}-{end})")
            #             found = True
            #             break
            #     if not found:
            #         print(f"[MISS] ID '{value}' not found.")
            # else:
            match = re.search(re.escape(value), text, re.IGNORECASE)
            if match:
                start, end = match.start(), match.end()
                print(f"[MATCH] {label} : '{value}' ({start}-{end})")
                entities.append((start, end, label_upper))
            else:
                print(f"[MISS] {label} '{value}' not found in OCR text.")

        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract") or \
                   doc.char_span(start, end, label=label, alignment_mode="expand")
            if not span:
                print(f"[WARN] Could not align span for: '{text[start:end]}' ({label})")
            else:
                spans.append(span)

        doc.ents = spans
        db.add(doc)

    db.to_disk(OUTPUT_FILE)
    print(f"\n[ok] Saved training data to '{OUTPUT_FILE}' with {len(logs)} documents.")

if __name__ == "__main__":
    generate_training_data()
