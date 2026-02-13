# Emotion-Based Break Recommendation System  
CNN–LSTM Video Emotion Classification | Flask Deployment  

🔗 Repository:  
https://github.com/Chenry513/Emotion-Break-Recommendation-System  

---

## Overview

This project builds an emotion-aware break recommendation system that analyzes gameplay video to detect sustained frustration or anger and recommend breaks when needed.

Rather than classifying individual frames independently, the system models emotional patterns over time using temporal deep learning.

---

## Problem Motivation

Emotion detection from video is a temporal problem:

- Emotions evolve across frames  
- Single-frame classifiers ignore context  
- Sustained frustration matters more than isolated expressions  

We initially explored DeepFace for frame-level emotion scoring, but it does not model temporal structure. To ensure a fair comparison, we built two full video pipelines.

---

## Model Pipelines

### Baseline: OpenFace + LSTM
- Extract facial Action Units (AUs) per frame using OpenFace  
- Feed AU sequences into an LSTM  
- Captures temporal facial muscle dynamics  

### Main Model: CNN → LSTM
- Use MobileNetV2 to extract spatial features from raw frames  
- Pass features into an LSTM  
- Learns visual emotion patterns end-to-end  

---

## Results

- Binary classification: Anger vs Neutral  
- Accuracy: ~70%  
- F1 Score: 0.70  

The CNN → LSTM model performed slightly better due to learning richer visual representations directly from video frames.

---

## Project Structure

```
├── CMPT419 Dataset/
│   ├── Anger/anger mp4 files
│   ├── Neutral/neutral mp4 files
│   └── openface_output/ openface csv annotations
├── static/
│   └── styles/
│       └── SFU.jpg
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/
│   ├── Anger1.mp4
│   ├── Anger1.5.mp4
│   ├── Anger1m.mp4
│   ├── Neutral1.mp4
│   ├── Neutral1.4.mp4
│   └── Neutral1m.mp4
├── app.py
├── baseline_model.ipynb
├── cnn_lstm_anger_classifier.h5
├── cnn_lstm_model.ipynb
├── readme
└── requirements.txt
```

---

## Setup Instructions

1. Ensure Python version is 3.10  
2. Install dependencies:

pip install -r requirements.txt  

3. Run the app:

python app.py  

4. Open your browser and go to:

http://127.0.0.1:5000  

Drag and drop sample .mp4 files from the uploads/ folder to test.

---

## Final Notes

This project focuses on modeling emotional dynamics over time rather than relying on isolated frame predictions.  

The primary contribution is the structured comparison between handcrafted facial features (OpenFace) and learned deep visual features (CNN–LSTM), along with full deployment into an interactive web application.

