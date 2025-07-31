# Face Similarity Comparison Tool 🧑‍🤝‍🧑🔍

https://face-similarity-tool.streamlit.app/

This project is a face comparison tool designed to analyze the similarity between faces detected in two uploaded images.
It uses the `face_recognition` library for facial detection and encoding, and `Streamlit` for an interactive web interface.

## Features

- Upload two face images and compare them
- Handles errors if no face is detected
- Shows face embeddings and distance
- Provides result as: "Same Person", "Likely Same", or "Different People"
- Visual result display side-by-side

## Requirements

Install system dependency:

```bash
sudo apt-get install libgl1-mesa-glx
```

Then install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```
