import streamlit as st
import numpy as np
import cv2
import spacy
import subprocess
import sys
import os
import sqlite3
import json
import fitz

from pdf2image import convert_from_bytes
from PIL import Image
from utils import (
    preprocess_image,
    extract_text,
    extract_text_gcv,
    parse_fields,
    parse_with_spacy,
    save_correction,
    get_nlp,
    update_model_on_correction
)

DB_FILE = "corrections.db"

st.set_page_config(page_title="CuriousScanner", layout="centered")
st.title("🤖 CuriousScanner")
st.markdown("Upload or capture a document, and I’ll extract the details dynamically!")

uploaded_file = st.file_uploader("📂 Upload Document", type=["jpg", "jpeg", "png", "pdf"])

if not uploaded_file:
    uploaded_file = st.camera_input("📸 Or capture using webcam")

ocr_engine = st.radio("🔍 Choose OCR Engine", ["Tesseract (Offline)", "Google Cloud Vision (GCV)"], horizontal=True)
mode = st.radio("🧠 Choose Parsing Mode", ["AI-powered (NER)", "Regex (Classic)"], horizontal=True)

if uploaded_file:
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()

        try:
            with st.spinner("🔍 Extracting text from PDF..."):
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text().strip()
        except Exception as e:
            st.error(f"Failed to open PDF: {e}")
            st.stop()

        if not text.strip():
            st.warning("⚠️ No extractable text found. Falling back to OCR...")
            try:
                with st.spinner("🖼️ Converting PDF to image for OCR..."):
                    images = convert_from_bytes(pdf_bytes)
            except Exception as e:
                st.error(f"PDF to image conversion failed: {e}")
                st.stop()

            if not images:
                st.error("❌ No pages found in PDF.")
                st.stop()

            image = images[0]
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.image(image, caption="PDF Page (Rendered as Image)", use_container_width=True)

            if ocr_engine == "Google Cloud Vision (GCV)":
                with st.spinner("📡 Using GCV for OCR..."):
                    # text = extract_text_gcv(np.array(image).tobytes())
                    _, buffer = cv2.imencode(".jpg", np.array(image))
                    text = extract_text_gcv(buffer.tobytes())
            else:
                use_adaptive = st.checkbox("🧪 Use Adaptive Thresholding", value=False)
                processed_img = preprocess_image(img_cv, use_adaptive=use_adaptive)
                text = extract_text(processed_img)
        else:
            st.success("✅ Extracted text directly from PDF! (No OCR needed)")
            image = None
    else:
        image = Image.open(uploaded_file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if ocr_engine == "Google Cloud Vision (GCV)":
            with st.spinner("📡 Using GCV for OCR..."):
                # text = extract_text_gcv(np.array(image).tobytes())
                _, buffer = cv2.imencode(".jpg", np.array(image))
                text = extract_text_gcv(buffer.tobytes())
        else:
            use_adaptive = st.checkbox("🧪 Use Adaptive Thresholding", value=False)
            processed_img = preprocess_image(img_cv, use_adaptive=use_adaptive)
            text = extract_text(processed_img)

    if text.strip():
        text = st.text_area("📜 OCR Text Output (Editable)", text, height=150)

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

        cursor.execute("SELECT DISTINCT doc_type FROM corrections")
        doc_types = [row[0] for row in cursor.fetchall()]
        conn.close()

        selected_doc_type = st.selectbox("Select Document Type", doc_types + ["➕ New Type"])
        if selected_doc_type == "➕ New Type":
            new_type = st.text_input("Enter New Doc Type Name")
            if new_type:
                doc_type = new_type.strip()
            else:
                doc_type = ""
        else:
            doc_type = selected_doc_type

        if not doc_type:
            st.error("❌ Please enter or select a valid document type.")
            st.stop()

        MODEL_DIR = f"ner_model/{doc_type}/model-best"

        if mode == "AI-powered (NER)":
            if not os.path.exists(os.path.join(MODEL_DIR, "meta.json")):
                st.warning("⚠️ NER model not found yet. Please save a correction or click 'Retrain' to initialize.")
                fields = {}
            else:
                fields = parse_with_spacy(text, model_path=MODEL_DIR)
        else:
            fields = parse_fields(text)

        if "custom_fields" not in st.session_state:
            st.session_state.custom_fields = []

        st.subheader("✏️ Verify & Correct Extracted Fields")
        corrected = {}

        st.markdown("#### 🔍 Extracted Fields")
        for key in sorted(fields.keys()):
            corrected[key] = st.text_input(f"{key}", value=fields[key])

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
            save_correction(text, corrected, doc_type=doc_type, predicted_data=fields)

            if not os.path.exists(os.path.join(MODEL_DIR, "meta.json")):
                st.info("🧠 Model not found — training from scratch now...")
                python_path = sys.executable
                result = subprocess.run(
                    [python_path, "generate_spacy_data.py", "--doc-type", doc_type],
                    cwd=os.getcwd(),
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True
                )
                with st.expander("📦 Auto-training Output"):
                    st.code(result.stdout)
                    if result.returncode != 0:
                        st.error("❌ Auto-training failed")
                        st.code(result.stderr)
                        st.stop()
                st.success("✅ Model trained successfully!")

            try:
                nlp = spacy.load(MODEL_DIR)
            except Exception as e:
                st.error(f"❌ Failed to load trained model: {e}")
                st.stop()

            losses = update_model_on_correction(nlp, text, corrected)
            nlp.to_disk(MODEL_DIR)

            st.success(f"📁 Model updated and saved to {MODEL_DIR}")
            st.info(f"🤖 Model updated with correction. Losses: {losses}")

            st.markdown("---")
            st.caption("📚 This scanner improves the more you use it. Feedback = better AI.")
    else:
        st.warning("⚠️ OCR failed. Try again with better lighting or focus.")

st.markdown("---")
st.subheader("🔁 Retrain NER Model")
use_latest_only = st.checkbox("Train on latest correction only", value=False)

if st.button("🚀 Retrain Now"):
    if 'doc_type' not in locals():
        st.error("⚠️ Please select a document type first.")
        st.stop()

    python_path = sys.executable
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    args = [python_path, "generate_spacy_data.py", "--doc-type", doc_type]
    if use_latest_only:
        args.append("--latest-only")

    with st.spinner("🧠 Training NER model..."):
        result = subprocess.run(
            args,
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
