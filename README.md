# Image Processing Exam

Rafif Dzakwan Muchsin (NIM: 41524010014)

Folder structure:
- Number-1-Histogram-Filtering-Domain
- Number-2-Face-Attendance-Haar-Cascade
- Number-3-Interactive-Object-Detection

## Overview

**Image Enhancement & Noise Filtering**  
This module focuses on improving image quality and analyzing pixel intensity distributions.

- Histogram Analysis & Equalization: Used to visualize pixel intensity distributions and enhance image contrast through global histogram equalization.
- Noise Modeling & Reduction: Simulates Gaussian noise and salt-and-pepper noise, followed by spatial filtering techniques such as mean (averaging) filters, Gaussian filters, and median filters to restore image clarity and reduce noise artifacts.

**Face Recognition Attendance System**  
A lightweight attendance system based on classical computer vision techniques.

- Face Detection: Employs Haar Cascade classifiers to detect human faces in images or uploaded video files.
- Face Recognition: Uses the Local Binary Pattern Histogram (LBPH) algorithm, trained from reference face images, to compare facial features and determine user presence.
- Attendance Decision: Presence is determined by comparing histogram distance values against a predefined threshold (`THRESHOLD = 60`).

**Interactive Object Detection**  
An interactive computer vision application for object identification using a pretrained deep learning model.

- Object Detection: Utilizes a pretrained MobileNet-SSD model to detect common objects in images or video based on confidence scores.
- Visualization: Displays bounding boxes and class labels interactively, allowing users to explore detected objects and adjust detection sensitivity.

### Usage: Face Attendance System

![Face Attendance System](./Number-2-Face-Attendance-Haar-Cascade/Face%20Attendance%20System.png)

## Deployment

```bash
# Python 3.9 – 3.12
pip install -r requirements.txt

# Run notebook cells for 1 and 3.
# Follow the instructions below for 2.

cd Number-2-Face-Attendance-Haar-Cascade
streamlit run app.py
```
