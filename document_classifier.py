# document_classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os

MODEL_PATH = "classifier_model.pkl"


def train_classifier():
    INPUT_FILE = "corrections_log.json"
    MODEL_PATH = "classifier_model.pkl"

    if not os.path.exists(INPUT_FILE):
        print(f"[warn] No corrections file found at {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    texts = []
    labels = []

    for entry in logs:
        text = entry.get("ocr_text", "").strip()
        label = entry.get("doc_type", "").strip().lower()

        if text and label:
            texts.append(text)
            labels.append(label)

    if not texts:
        print("[warn] No labeled data found in corrections log.")
        return

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, labels)

    joblib.dump((vec, clf), MODEL_PATH)
    print(f"[ok] Retrained document classifier with {len(texts)} samples.")

def predict_doc_type(text):
    vec, clf = joblib.load(MODEL_PATH)
    return clf.predict(vec.transform([text]))[0]


# 🔁 Add this
if __name__ == "__main__":
    train_classifier()

