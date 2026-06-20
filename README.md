# Factory Compliance System

## Overview

Factory Compliance System is an AI-powered workplace safety monitoring application that analyzes factory videos and detects safe or unsafe behaviors.

The system uses a custom-trained ResNet18 deep learning model built with PyTorch. Since training directly on videos is computationally expensive, the video classification problem was converted into an image classification problem by extracting frames from videos.

The application allows users to upload an MP4 video through a Streamlit interface. The backend processes the video, extracts frames, runs inference using the trained model, and returns a final Safe or Unsafe prediction.

---

## Features

* Upload factory surveillance videos
* Automatic frame extraction from videos
* AI-based safety compliance detection
* FastAPI backend for model inference
* Streamlit frontend for user interaction
* Custom-trained ResNet18 model
* Lightweight deployment suitable for CPU environments

---

## Project Pipeline

Dataset (Kaggle Safe & Unsafe Behaviours Dataset)

↓

Select a balanced subset of videos

↓

Extract 5 frames per video

↓

Create image dataset

↓

Train ResNet18 model

↓

Save model.pth

↓

FastAPI backend

↓

Streamlit frontend

↓

Final Safe / Unsafe prediction

---

## Tech Stack

* Python
* PyTorch
* ResNet18
* OpenCV
* FastAPI
* Streamlit

---

## Model Training

Training strategy:

* Video classification converted into image classification
* Extracted 5 frames from each selected video
* Trained using ResNet18 transfer learning
* Binary classification:

  * Safe
  * Unsafe

Performance:

* Validation Accuracy: ~80%

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn src.api:app --reload --port 8000 
```

Start Streamlit:

```bash
streamlit run src\dashboard\app.py
```

Open:

```
http://localhost:8501
```

Upload an MP4 video and view the prediction.

---

## Future Improvements

* Real-time CCTV monitoring
* Multi-class safety violation detection
* Temporal models (LSTM/3D CNN)
* Alert notification system
* Dashboard analytics

---

## Author

Malik Ahsan Nasar 