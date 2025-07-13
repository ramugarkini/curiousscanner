import streamlit as st
import numpy as np
import cv2
import spacy
import subprocess
import sys
import os
import sqlite3
import json

from PIL import Image
from utils import (
    preprocess_image,
    extract_text,
    parse_fields,
    parse_with_spacy,
    save_correction,
    get_nlp,
    update_model_on_correction
)
from document_classifier import predict_doc_type

DB_FILE = "corrections.db"

st.set_page_config(page_title="CuriousScanner", layout="centered")

st.title("🤖 CuriousScanner")
st.markdown("Upload or capture a **college ID card**, and I’ll extract the details dynamically!")

# File uploader or camera input
uploaded_file = st.file_uploader("📂 Upload ID Card Image", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    uploaded_file = st.camera_input("📸 Or capture ID card using webcam")

# Parsing mode selection
mode = st.radio("🧠 Choose Parsing Mode", ["AI-powered (NER)", "Regex (Classic)"], horizontal=True)

if uploaded_file:
    # Load and convert image
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Show image
    st.image(uploaded_file, caption="ID Card", use_container_width=True)

    # OCR and preprocessing
    # st.info("🔍 Processing image...")
    use_adaptive = st.checkbox("🧪 Use Adaptive Thresholding", value=False)
    processed_img = preprocess_image(img_cv, use_adaptive=use_adaptive)

    text = extract_text(processed_img)
    doc_type = predict_doc_type(text)
    st.info(f"📄 Detected document type: {doc_type.title()}")
    st.text(f"[DEBUG] OCR text length: {len(text)}")

    if text.strip():
        st.text_area("📜 OCR Text Output", text, height=150)
    else:
        st.warning("⚠️ OCR failed. Try again with better lighting or focus.")

    # Choose parsing method
    if mode == "AI-powered (NER)":
        doc_model_map = {
            "student": "ner_model/student",
            "employee": "ner_model/employee",
            "invoice": "ner_model/invoice",
            "visiting_card": "ner_model/visiting_card"
        }
        model_dir = f"ner_model/{doc_type}/model-best"
        if not os.path.exists(model_dir):
            model_dir = "ner_model/model-best"

        fields = parse_with_spacy(text, model_path=model_dir)

    else:
        fields = parse_fields(text)

    # Session state for custom field additions
    if "custom_fields" not in st.session_state:
        st.session_state.custom_fields = []

    st.subheader("✏️ Verify & Correct Extracted Fields")
    corrected = {}

    st.markdown("#### 🔍 Extracted Fields")
    for key in sorted(fields.keys()):
        corrected[key] = st.text_input(f"{key}", value=fields[key])

    # ➕ Add custom fields
    st.markdown("#### ➕ Add Custom Field")
    with st.form("add_field_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([4, 4, 2])
        new_key = col1.text_input("Field Name", key="new_key")
        new_value = col2.text_input("Field Value", key="new_value")
        col3.markdown(" ")
        col3.markdown(" ")
        add_btn = col3.form_submit_button("➕ Add Field")

        if add_btn and new_key:
            st.session_state.custom_fields.append((new_key, new_value))

    # st.markdown("#### 📝 Custom Fields")
    delete_indices = []
    for i, (key, value) in enumerate(st.session_state.custom_fields):
        cols = st.columns([3, 3, 1])
        updated_key = cols[0].text_input(f"Key {i+1}", key, key=f"custom_key_{i}")
        updated_val = cols[1].text_input(f"Value {i+1}", value, key=f"custom_val_{i}")
        cols[2].markdown(" ")
        cols[2].markdown(" ")
        delete = cols[2].button("🗑️", key=f"delete_btn_{i}")
        if delete:
            delete_indices.append(i)
        else:
            corrected[updated_key] = updated_val

    for i in sorted(delete_indices, reverse=True):
        del st.session_state.custom_fields[i]

    if st.button("✅ Save Correction"):
        save_correction(text, corrected, predicted_data=fields, doc_type=doc_type)
        # st.success("✅ Correction saved to SQLite!")

        nlp = spacy.load(model_dir) if os.path.exists(model_dir) else None
        if nlp:
            losses = update_model_on_correction(nlp, text, corrected)
            nlp.to_disk(model_dir)
            st.success(f"📁 Model updated and saved to {model_dir}")
            st.info(f"🤖 Model updated with correction. Losses: {losses}")
        else:
            st.warning("⚠️ Model not loaded. Skipped online update.")

    st.markdown("---")
    st.caption("📚 This scanner improves the more you use it. Feedback = better AI.")

# Retraining block
st.markdown("---")
st.subheader("🔁 Retrain NER Model")
st.write("Click the button below to regenerate training data and retrain the AI model based on saved corrections.")

if st.button("🚀 Retrain Now"):
    python_path = sys.executable
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with st.spinner("🧠 Training NER model (latest correction only)..."):
        result = subprocess.run(
            [python_path, "generate_spacy_data.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            env=env
        )
        with st.expander("📦 Training Output"):
            if result.returncode != 0:
                st.error("❌ Training failed")
                st.code(result.stderr)
            else:
                st.code(result.stdout)

    get_nlp.clear()

    # ✅ Show latest correction info used in training (from SQLite)
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT ocr_text, corrected_fields, reward FROM corrections ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            ocr_text, corrected_json, reward = row
            with st.expander("📘 Last Used Correction Data"):
                st.markdown("**📝 OCR Text Used for Training**")
                st.code(ocr_text, language="text")
                st.markdown("**✅ Corrected Fields Used for Training**")
                st.json(json.loads(corrected_json))

            st.subheader("🎯 Last Correction Reward")
            st.metric(label="Reward", value=f"{reward:.1f}")
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch correction from DB: {e}")

    st.success("✅ Retraining complete! Please refresh the app to load the new model.")
