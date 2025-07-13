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
)

st.set_page_config(page_title="CuriousScanner", layout="centered")

st.title("🤖 CuriousScanner")
st.markdown("Upload a **college ID card photo**, and I’ll extract the details dynamically!")

# File uploader
uploaded_file = st.file_uploader("Upload ID Card Image", type=["jpg", "jpeg", "png"])

# Mode toggle: AI vs Regex
mode = st.radio("🧠 Choose Parsing Mode", ["AI-powered (NER)", "Regex (Classic)"], horizontal=True)

if uploaded_file:
    # Safe image load and conversion
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Show image
    st.image(uploaded_file, caption="Uploaded ID Card", use_container_width=True)

    # OCR processing
    st.info("🔍 Processing image...")
    processed_img = preprocess_image(img_cv, use_adaptive=False)
    text = extract_text(processed_img)

    if text.strip():
        st.text_area("📜 OCR Text Output", text, height=150)
    else:
        st.warning("⚠️ OCR failed. Try a clearer image or toggle preprocessing settings.")

    # Use selected parsing mode
    if mode == "AI-powered (NER)":
        fields = parse_with_spacy(text)
    else:
        fields = parse_fields(text)

    st.subheader("✏️ Verify & Correct Extracted Fields")
    name = st.text_input("Name", fields.get("Name", ""))
    dob = st.text_input("Date of Birth", fields.get("DOB", ""))
    id_number = st.text_input("ID", fields.get("ID", ""))

    if st.button("✅ Save Correction"):
        corrected = {"Name": name, "DOB": dob, "ID": id_number}
        save_correction(text, corrected)
        st.success("✅ Saved successfully to corrections_log.json!")

    st.markdown("---")
    st.caption("📚 This demo improves over time with your feedback. CuriousScanner learns from you!")

import subprocess
import sys

st.markdown("---")
st.subheader("🔁 Retrain NER Model")
st.write("Click the button below to regenerate training data and retrain the NER model based on saved corrections.")

if st.button("🚀 Retrain Now"):
    python_path = sys.executable  # dynamically use current Python path

    with st.spinner("Generating training data..."):
        result1 = subprocess.run([python_path, "generate_spacy_data.py"], capture_output=True, text=True)
        st.code(result1.stdout or result1.stderr)

    with st.spinner("Training spaCy model..."):
        result2 = subprocess.run(
            [python_path, "-m", "spacy", "train", "config.cfg", "--output", "ner_model"],
            capture_output=True, text=True
        )
        st.code(result2.stdout or result2.stderr)

    st.success("✅ Retraining complete! Please **refresh** the app to load the new model.")

