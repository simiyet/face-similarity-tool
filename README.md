# Face Similarity Comparison Tool 👤🔍

This project is a face comparison tool designed to analyze the similarity between faces detected in one or more uploaded images. It uses the `face_recognition` library for facial detection and encoding, and `Streamlit` for an interactive web interface.

## Features

- 📂 Upload multiple images at once
- 👁‍🗨 Automatically detect faces and draw bounding boxes
- 🧰 Compare every detected face pair and calculate similarity score
- 🔢 Classify results based on thresholds: Same Person, Probably Similar, or Different Person
- 🌎 Browser-based interactive interface using Streamlit

## How It Works

1. The user uploads one or more images.
2. All faces in the images are detected and encoded.
3. Each pair of faces is compared using Euclidean distance.
4. Verdicts are determined using these thresholds:
   - Distance < 0.6 → Same Person
   - 0.6 ≤ Distance < 0.75 → Probably Similar
   - Distance ≥ 0.75 → Different Person
5. Results and marked-up face images are displayed visually.

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Required Packages

```txt
streamlit
face_recognition
numpy
pillow
matplotlib
```

## Suggested Tags

- face-recognition
- streamlit
- computer-vision
- similarity-matching
- python-app
- image-processing

## License

MIT License