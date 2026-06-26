🍌 Banana Ripeness Classification using YOLOv8
📌 Project Overview

This project classifies banana ripeness into four categories using a YOLOv8 Classification model.

The application allows users to upload an image of a banana and instantly predicts its ripeness stage.

Classes
🍌 Overripe
🍌 Ripe
🍌 Rotten
🍌 Unripe
Deep Learning Models Compared
Model	Validation Accuracy	Test Accuracy
Custom CNN	94.12%	93.42%
MobileNetV2	97.24%	96.98%
EfficientNetB0	34.55%	32.92%
ResNet50	98.31%	97.51%
YOLOv8 Classification	98.93%	Validation Evaluated
Technologies Used
Python
Streamlit
Ultralytics YOLOv8
PyTorch
NumPy
Pillow
Features
Upload banana images
Real-time ripeness prediction
Confidence score
Class probabilities
Modern web interface
Run Locally
pip install -r requirements.txt
streamlit run app.py
Deployment

This project is deployed using Streamlit Community Cloud.

Author

Naveen Kumar