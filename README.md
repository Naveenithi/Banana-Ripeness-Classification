🍌 Banana Ripeness Classification using Deep Learning and YOLOv8
📌 Project Overview

Banana ripeness assessment is an important task in agriculture, food quality inspection, supply chain management, and retail. Manual inspection is subjective and time-consuming. This project leverages Deep Learning and Computer Vision to automatically classify banana ripeness into four maturity stages using image classification models.

The project compares multiple deep learning architectures and deploys the best-performing model using a Streamlit web application for real-time prediction.

🎯 Objectives
Build an end-to-end banana ripeness classification system.
Compare the performance of multiple deep learning models.
Deploy the best-performing model using Streamlit.
Perform real-time image classification through a user-friendly web interface.
🗂️ Dataset

The dataset was obtained from Roboflow Universe and contains banana images categorized into four ripeness classes.

Classes
🍌 Overripe
🍌 Ripe
🍌 Rotten
🍌 Unripe

The dataset was divided into:

Training Set
Validation Set
Test Set

Image size:

224 × 224 pixels
🛠️ Technologies Used
Python
TensorFlow / Keras
PyTorch
Ultralytics YOLOv8
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Streamlit
🔬 Deep Learning Models Evaluated
Model	Validation Accuracy	Test Accuracy
Custom CNN	94.12%	93.42%
MobileNetV2	97.24%	96.98%
EfficientNetB0	34.55%	32.92%
ResNet50	98.31%	97.51%
YOLOv8 Classification	98.93% (Top-1 Validation Accuracy)	Deployed Model
📊 Project Workflow

Dataset Collection

⬇️

Data Preprocessing

⬇️

Image Augmentation

⬇️

Exploratory Data Analysis (EDA)

⬇️

Model Development

Custom CNN
MobileNetV2
EfficientNetB0
ResNet50
YOLOv8 Classification

⬇️

Model Evaluation

⬇️

Streamlit Deployment

🚀 Features
Upload banana images
Real-time banana ripeness prediction
Confidence score visualization
Comparison of multiple deep learning models
Interactive Streamlit interface
🌐 Live Demo

Streamlit Application

https://banana-ripeness-classification-vyesp3sfwpchtedbcpuvvj.streamlit.app/

📁 Project Structure
Banana-Ripeness-Classification/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
├── notebooks/
├── models/
├── images/
└── screenshots/
⚙️ Installation

Clone the repository and install the required packages.

pip install -r requirements.txt

Run the application.

streamlit run app.py
📈 Results

Among all evaluated models, YOLOv8 Classification achieved the highest validation accuracy and was selected for deployment because of its excellent prediction performance and inference speed.

The project demonstrates that transfer learning and modern deep learning architectures can accurately classify banana ripeness with high reliability.

🔮 Future Enhancements

Future improvements include:

YOLOv8 Object Detection for banana localization.
Automatic banana counting.
Camera-based real-time inference.
Mobile application deployment.
Cloud-based inference API.
👨‍💻 Author

Naveen Kumar

Finance & Operations Professional transitioning into Data Science

Skilled in Python, SQL, Machine Learning, Deep Learning, Computer Vision, and Streamlit.

⭐ Acknowledgement

This project was developed as part of the GUVI – HCL Data Science & Machine Learning Program, demonstrating an end-to-end Deep Learning workflow from data preparation and model training to deployment.
