import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import (
    preprocess_image,
    extract_text,
    parse_fields,
    parse_with_spacy,
    save_correction,
    get_nlp  
)
import subprocess
import sys
import os
import json

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
    st.info("🔍 Processing image...")
    use_adaptive = st.checkbox("🧪 Use Adaptive Thresholding", value=False)
    processed_img = preprocess_image(img_cv, use_adaptive=use_adaptive)

    text = extract_text(processed_img)
    st.text(f"[DEBUG] OCR text length: {len(text)}")

    if text.strip():
        st.text_area("📜 OCR Text Output", text, height=150)
    else:
        st.warning("⚠️ OCR failed. Try again with better lighting or focus.")

    # Choose parsing method
    if mode == "AI-powered (NER)":
        fields = parse_with_spacy(text)
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
        col1, col2, col3 = st.columns([4, 4, 2])  # Adjust width ratio as needed
        new_key = col1.text_input("Field Name", key="new_key")
        new_value = col2.text_input("Field Value", key="new_value")
        col3.markdown(" ")
        col3.markdown(" ")
        add_btn = col3.form_submit_button("➕ Add Field")

        if add_btn and new_key:
            st.session_state.custom_fields.append((new_key, new_value))



    st.markdown("#### 📝 Custom Fields")

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

    # if st.button("🧹 Clear Custom Fields"):
    #     st.session_state.custom_fields = []

    if st.button("✅ Save Correction"):
        save_correction(text, corrected)
        st.success("✅ Correction saved to `corrections_log.json`!")

        # Show latest saved correction
        if os.path.exists("corrections_log.json"):
            with open("corrections_log.json", "r", encoding="utf-8") as f:
                corrections = json.load(f)
                if corrections:
                    latest = corrections[-1]
                    st.markdown("#### 📝 Last Saved OCR Text")
                    st.code(latest["ocr_text"], language="text")
                    st.markdown("#### ✅ Last Saved Fields")
                    st.json(latest["corrected_fields"])

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

    with st.spinner("🔧 Generating training data..."):
        result1 = subprocess.run(
            [python_path, "generate_spacy_data.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            env=env  # ✅ injects UTF-8 encoding into subprocess
        )
        with st.expander("📄 Training Data Output"):
            if result1.returncode != 0:
                st.error("❌ Failed to run generate_spacy_data.py")
                st.code(result1.stderr)
            else:
                st.code(result1.stdout)


    with st.spinner("🧠 Training NER model..."):
        result2 = subprocess.run(
            [python_path, "-m", "spacy", "train", "config.cfg", "--output", "ner_model"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            env=env
        )
        with st.expander("📦 Training Output"):
            if result2.returncode != 0:
                st.error("❌ Failed to run generate_spacy_data.py")
                st.code(result2.stderr)
            else:
                st.code(result2.stdout)

    get_nlp.clear()  # ✅ Reload new model


    # ✅ Show latest correction info used in training
    if os.path.exists("corrections_log.json"):
        with open("corrections_log.json", "r", encoding="utf-8") as f:
            corrections = json.load(f)
            if corrections:
                latest = corrections[-1]
                with st.expander("📘 Last Used Correction Data"):
                    st.markdown("**📝 OCR Text Used for Training**")
                    st.code(latest["ocr_text"], language="text")

                    st.markdown("**✅ Corrected Fields Used for Training**")
                    st.json(latest["corrected_fields"])

    st.success("✅ Retraining complete! Please refresh the app to load the new model.")
