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
import subprocess
import sys

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

    # Dynamic field input
    st.subheader("✏️ Verify & Correct Extracted Fields")
    corrected = {}
    for key, value in fields.items():
        corrected[key] = st.text_input(key, value)

    if st.button("✅ Save Correction"):
        save_correction(text, corrected)
        st.success("✅ Correction saved to `corrections_log.json`!")

    st.markdown("---")
    st.caption("📚 This scanner improves the more you use it. Feedback = better AI.")

# Retraining block
st.markdown("---")
st.subheader("🔁 Retrain NER Model")
st.write("Click the button below to regenerate training data and retrain the AI model based on saved corrections.")

if st.button("🚀 Retrain Now"):
    python_path = sys.executable  # Use current Python interpreter

    with st.spinner("🔧 Generating training data..."):
        result1 = subprocess.run([python_path, "generate_spacy_data.py"], capture_output=True, text=True)
        st.code(result1.stdout or result1.stderr)

    with st.spinner("🧠 Training NER model..."):
        result2 = subprocess.run(
            [python_path, "-m", "spacy", "train", "config.cfg", "--output", "ner_model"],
            capture_output=True, text=True
        )
        st.code(result2.stdout or result2.stderr)

    st.success("✅ Retraining complete! Please refresh the app to load the new model.")
