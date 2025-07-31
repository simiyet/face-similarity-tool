import face_recognition
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

def load_and_encode_image(image):
    image_np = np.array(image)
    locations = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, locations)
    return encodings, locations

def draw_faces(image, face_locations):
    image_pil = image.copy()
    draw = ImageDraw.Draw(image_pil)
    for top, right, bottom, left in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
    return image_pil

def compare_faces(encodings, threshold_strict=0.6, threshold_loose=0.75):
    results = []
    for i in range(len(encodings)):
        for j in range(i+1, len(encodings)):
            dist = np.linalg.norm(encodings[i] - encodings[j])
            similarity = 1 - dist
            if dist < threshold_strict:
                verdict = "Same Person"
            elif dist < threshold_loose:
                verdict = "Probably Similar"
            else:
                verdict = "Different Person"
            results.append({
                'pair': (i+1, j+1),
                'distance': round(dist, 4),
                'similarity': round(similarity, 4),
                'verdict': verdict
            })
    return results

st.set_page_config(page_title="Face Similarity Tool", layout="wide")
st.title("👤 Face Similarity Comparison Tool")
st.markdown("Upload one or more images to compare the detected faces.")

uploaded_files = st.file_uploader("Upload Image(s)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    all_encodings = []
    all_labels = []
    all_faces_marked = []

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        encodings, locations = load_and_encode_image(image)

        for face_idx, enc in enumerate(encodings):
            all_encodings.append(enc)
            label = f"{uploaded_file.name} - Face {face_idx+1}"
            all_labels.append(label)

        marked_image = draw_faces(image, locations)
        st.image(marked_image, caption=f"{uploaded_file.name} - Faces detected", use_column_width=True)

    if len(all_encodings) >= 2:
        results = compare_faces(all_encodings)
        st.subheader("🔍 Comparison Results")
        for result in results:
            st.markdown(f"**{all_labels[result['pair'][0]-1]} <> {all_labels[result['pair'][1]-1]}**")
            st.text(f"Distance: {result['distance']} | Similarity: {result['similarity']*100:.2f}% | Result: {result['verdict']}")
    else:
        st.warning("At least two faces are required for comparison.")