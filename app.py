import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np
import tempfile
import os

def verify_faces(img1, img2):
    result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=True)
    return result["verified"], result["distance"]

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def main():
    st.title("Face Similarity Checker")
    st.write("Upload two images. Each must contain exactly one face.")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    with col2:
        uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and uploaded_file2:
        path1 = save_uploaded_file(uploaded_file1)
        path2 = save_uploaded_file(uploaded_file2)

        st.image(Image.open(path1), caption="Image 1", use_column_width=True)
        st.image(Image.open(path2), caption="Image 2", use_column_width=True)

        try:
            verified, distance = verify_faces(path1, path2)
            if verified:
                st.success(f"Faces Match! Distance: {distance:.4f}")
            else:
                st.error(f"Faces Do NOT Match. Distance: {distance:.4f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

        # Clean up
        os.remove(path1)
        os.remove(path2)

if __name__ == "__main__":
    main()
