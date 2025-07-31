import streamlit as st
import numpy as np
from PIL import Image
import face_recognition
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("🧑‍🤝‍🧑 Face Similarity Comparison Tool")

def load_and_detect_face(image_file, label):
    image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return None, f"No face detected in {label}."
    elif len(face_locations) > 1:
        return None, f"More than one face detected in {label}."
    else:
        encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        return (image, face_locations[0], encoding), None

def draw_face_box(image_np, face_location):
    top, right, bottom, left = face_location
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    rect = plt.Rectangle((left, top), right - left, bottom - top,
                         fill=False, color="lime", linewidth=2)
    ax.add_patch(rect)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

uploaded_files = st.file_uploader("Upload exactly 2 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_files[0], caption="Image 1", use_container_width=True)
    with col2:
        st.image(uploaded_files[1], caption="Image 2", use_container_width=True)

    data1, err1 = load_and_detect_face(uploaded_files[0], "Image 1")
    data2, err2 = load_and_detect_face(uploaded_files[1], "Image 2")

    if err1:
        st.error(err1)
    elif err2:
        st.error(err2)
    else:
        img1_np, loc1, enc1 = data1
        img2_np, loc2, enc2 = data2

        distance = np.linalg.norm(enc1 - enc2)
        st.markdown(f"### Distance: `{distance:.4f}`")

        if distance < 0.4:
            result = "✅ Same Person"
        elif distance < 0.6:
            result = "🟡 Likely Same Person"
        else:
            result = "❌ Different People"

        st.subheader(f"Result: {result}")

        st.markdown("#### Face Detection Visualization")
        col3, col4 = st.columns(2)
        with col3:
            st.image(draw_face_box(img1_np, loc1), caption="Image 1 Detected Face", use_container_width=True)
        with col4:
            st.image(draw_face_box(img2_np, loc2), caption="Image 2 Detected Face", use_container_width=True)

else:
    st.info("Please upload exactly two image files.")