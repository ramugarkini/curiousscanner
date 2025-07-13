# document_classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import sqlite3
import json
import os

MODEL_PATH = "classifier_model.pkl"
DB_FILE = "corrections.db"

def train_classifier():
    if not os.path.exists(DB_FILE):
        print(f"[warn] No SQLite database found at {DB_FILE}")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT ocr_text, doc_type FROM corrections")
    rows = cursor.fetchall()
    conn.close()

    texts = []
    labels = []

    for text, label in rows:
        if text and label:
            texts.append(text.strip())
            labels.append(label.strip().lower())

    if not texts:
        print("[warn] No labeled data found in corrections database.")
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

# 🔁 Run training if executed directly
if __name__ == "__main__":
    train_classifier()
