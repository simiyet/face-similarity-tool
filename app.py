import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np

def verify_faces(img1, img2):
    result = DeepFace.verify(img1_path=np.array(img1), img2_path=np.array(img2), enforce_detection=True)
    return result

st.set_page_config(page_title="Face Similarity Tool", layout="centered")
st.title("👤 Face Similarity Comparison Tool")

st.markdown("Upload **exactly two** images. Each image must contain **one and only one** face.")

uploaded_files = st.file_uploader("Upload 2 Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 2:
        st.error("⚠️ Please upload exactly 2 images.")
    else:
        try:
            img1 = Image.open(uploaded_files[0]).convert("RGB")
            img2 = Image.open(uploaded_files[1]).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.image(img1, caption="Image 1", use_column_width=True)
            with col2:
                st.image(img2, caption="Image 2", use_column_width=True)

            st.subheader("🔍 Verifying Similarity...")
            result = verify_faces(img1, img2)

            distance = result.get("distance", 1.0)
            if result.get("verified"):
                st.success(f"✅ Same Person — Distance: {distance:.4f}")
            else:
                st.warning(f"❌ Different Person — Distance: {distance:.4f}")

        except ValueError as ve:
            error_str = str(ve).lower()
            if "no face" in error_str:
                st.error("🚫 One or both images do not contain a detectable face.")
            elif "more than one" in error_str:
                st.error("🚫 Each image must contain exactly one face. Multiple faces were detected.")
            else:
                st.error(f"❗ Unexpected error: {ve}")
