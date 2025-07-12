# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import preprocess_image, extract_text, parse_fields, save_correction

st.set_page_config(page_title="CuriousScanner3000", layout="centered")

st.title("🤖 CuriousScanner3000")
st.markdown("Upload a **college ID card photo**, and I’ll extract the details dynamically!")

uploaded_file = st.file_uploader("Upload ID Card Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Safe image load and conversion
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Show image
    st.image(uploaded_file, caption="Uploaded ID Card", use_container_width=True)

    # OCR process
    st.info("🔍 Processing image...")
    processed_img = preprocess_image(img_cv, use_adaptive=False)  # Use grayscale for now
    text = extract_text(processed_img)

    if text.strip():
        st.text_area("📜 OCR Text Output", text, height=150)
    else:
        st.warning("⚠️ OCR failed. Try a clearer image or toggle preprocessing settings.")

    fields = parse_fields(text)

    st.subheader("✏️ Verify & Correct Extracted Fields")
    name = st.text_input("Name", fields.get("Name", ""))
    dob = st.text_input("Date of Birth", fields.get("DOB", ""))
    id_number = st.text_input("ID Number", fields.get("ID Number", ""))

    if st.button("✅ Save Correction"):
        corrected = {"Name": name, "DOB": dob, "ID Number": id_number}
        save_correction(text, corrected)
        st.success("✅ Saved successfully to corrections_log.json!")

    st.markdown("---")
    st.caption("📚 This is a demo of a dynamic OCR parser that learns from corrections.")
