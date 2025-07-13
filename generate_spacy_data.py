import json
import re
import spacy
from spacy.tokens import DocBin

INPUT_FILE = "corrections_log.json"
OUTPUT_FILE = "training_data.spacy"

LABEL_MAP = {
    "Name": "NAME",
    "DOB": "DOB",
    "ID": "ID"
}

def generate_training_data():
    nlp = spacy.blank("en")
    db = DocBin()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    for entry in logs:
        text = entry["ocr_text"]
        corrected = entry["corrected_fields"]
        entities = []

        print(f"\n[ENTRY] OCR Text:\n{text.strip()}")

        for label, value in corrected.items():
            clean_label = LABEL_MAP.get(label, label.upper().replace(" ", "_"))

            if label == "ID":
                id_clean = re.sub(r"\D", "", value)
                found = False

                # Clean text for matching: collapse all whitespace/newlines to space
                normalized_text = re.sub(r"[\n\r]", " ", text)

                for match in re.finditer(r'\d{4}[\s\-]?\d{4}[\s\-]?\d{4}', normalized_text):
                    candidate = match.group(0)
                    candidate_clean = re.sub(r"\D", "", candidate)
                    if candidate_clean == id_clean:
                        # Map back to original text index
                        start = text.find(candidate)
                        if start != -1:
                            end = start + len(candidate)
                            print(f"[MATCH] ID → '{candidate}' ({start}-{end})")
                            entities.append((start, end, clean_label))
                            found = True
                            break
                if not found:
                    print(f"[MISS] ID '{value}' not found in OCR text.")


            else:
                match = re.search(re.escape(value), text, re.IGNORECASE)
                if match:
                    start, end = match.start(), match.end()
                    print(f"[MATCH] {label} → '{value}' ({start}-{end})")
                    entities.append((start, end, clean_label))
                else:
                    print(f"[MISS] {label} '{value}' not found in OCR text.")

        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract") \
                or doc.char_span(start, end, label=label, alignment_mode="expand")
            if not span:
                print(f"[WARN] Failed to align span: '{text[start:end]}' for label={label}")
            else:
                spans.append(span)

        doc.ents = spans
        db.add(doc)

    db.to_disk(OUTPUT_FILE)
    print(f"\n[ok] Saved training data to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_training_data()
