# Emotion-Detection
# Emotion-Detection
# Real-time Emotion Detection using OpenCV and DeepFace

This project captures video from a webcam, detects faces in real-time using OpenCV, and analyzes the emotions of detected faces using DeepFace. The dominant emotion and its corresponding percentage are displayed on the video feed.

## Features

- **Real-time Face Detection**: Uses OpenCV's pre-trained Haar Cascade classifier to detect faces in real-time.
- **Emotion Analysis**: Leverages DeepFace to analyze emotions from detected faces.
- **Live Feedback**: Displays the dominant emotion and its percentage on the video feed.

## Requirements

- Python 3.x
- OpenCV
- DeepFace

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/real-time-emotion-detection.git
   cd real-time-emotion-detection
   
2.**Install the required packages**:
     pip install opencv-python-headless deepface

## Usage
1. **Run the script**:
   python emotion_detection.py
   
2.**Interacting with the application**:
   1.The webcam will start, and the video feed will display the detected faces with rectangles.
   2.The dominant emotion and its percentage will appear above each detected face.
   3.Press q to exit the application
