import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np
import tempfile
import os

def verify_faces(img1, img2):
    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=True)
        return result
    except Exception as e:
        return {"error": str(e)}

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def main():
    st.set_page_config(page_title="Face Similarity Tool", layout="centered")
    st.title("🔍 Face Similarity Comparison")

    uploaded_files = st.file_uploader("Upload exactly 2 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 2:
        file_paths = []

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_files[0], caption="Image 1", use_container_width=True)
        with col2:
            st.image(uploaded_files[1], caption="Image 2", use_container_width=True)

        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            file_paths.append(file_path)

        with st.spinner('Analyzing...'):
            result = verify_faces(file_paths[0], file_paths[1])

        if "error" in result:
            st.error(f"Face verification failed: {result['error']}")
        else:
            distance = result.get("distance", 1.0)
            verified = result.get("verified", False)

            st.markdown("---")
            st.subheader("Result")
            if verified:
                st.success(f"✅ The faces match. Similarity Score: {1 - distance:.2f}")
            else:
                st.warning(f"❌ The faces do not match. Similarity Score: {1 - distance:.2f}")
    elif uploaded_files:
        st.warning("Please upload exactly 2 images.")

if __name__ == '__main__':
    main()
