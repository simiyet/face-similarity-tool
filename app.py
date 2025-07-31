import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from deepface import DeepFace
from deepface.detectors import FaceDetector

st.set_page_config(page_title="Face Similarity Checker", layout="centered")
st.title("🧑‍🤝‍🧑 Face Similarity Comparison Tool")

uploaded_files = st.file_uploader("Upload exactly 2 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def detect_single_face(image):
    try:
        img_array = np.array(image)
        detector_backend = 'retinaface'
        face_objs = FaceDetector.build_model(detector_backend)
        faces = FaceDetector.detect_faces(face_objs, detector_backend, img_array, align=False)
        if len(faces) != 1:
            return None
        return faces[0]["facial_area"]
    except:
        return None

def show_images_side_by_side(img1, img2):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Image 1", use_container_width=True)
    with col2:
        st.image(img2, caption="Image 2", use_container_width=True)

def verify_faces(img1, img2):
    result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=False, detector_backend="retinaface")
    return result

if uploaded_files and len(uploaded_files) == 2:
    try:
        img1 = Image.open(uploaded_files[0]).convert("RGB")
        img2 = Image.open(uploaded_files[1]).convert("RGB")

        face1 = detect_single_face(img1)
        face2 = detect_single_face(img2)

        if face1 is None or face2 is None:
            st.error("Each image must contain exactly one detectable face.")
        else:
            show_images_side_by_side(img1, img2)

            # Save temporarily to disk for DeepFace
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            path1 = os.path.join(temp_dir, "img1.jpg")
            path2 = os.path.join(temp_dir, "img2.jpg")
            img1.save(path1)
            img2.save(path2)

            with st.spinner("Analyzing faces..."):
                result = verify_faces(path1, path2)

            verified = result["verified"]
            distance = result["distance"]
            threshold = result["threshold"]
            model_used = result["model"]

            st.subheader("Comparison Result")
            st.markdown(f"**Verified:** {'✅ Match' if verified else '❌ No Match'}")
            st.markdown(f"**Distance:** {distance:.4f} (Threshold: {threshold:.4f})")
            st.markdown(f"**Model:** {model_used}")

    except Exception as e:
        st.error(f"Face verification failed: {str(e)}")
else:
    st.info("Please upload exactly 2 images.")
