import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import tempfile
import time
import os

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="🍌 Banana Ripeness Classification",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:#2E8B57;
}

.metric-container{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():

    model_path = "best.pt"

    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.pt' not found.")
        st.stop()

    return YOLO(model_path)

model = load_model()

CLASS_NAMES = model.names

# ==========================================================
# HELPER FUNCTION
# ==========================================================
def predict_image(image):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:

        image.save(temp.name)

        start = time.time()

        results = model.predict(
            source=temp.name,
            verbose=False
        )

        end = time.time()

    probs = results[0].probs.data.cpu().numpy()

    pred = int(np.argmax(probs))

    confidence = float(probs[pred])

    inference = end - start

    return pred, confidence, inference, probs

# ==========================================================
# MODEL COMPARISON DATA
# ==========================================================
comparison_df = pd.DataFrame({

    "Model":[
        "Custom CNN",
        "MobileNetV2",
        "EfficientNetB0",
        "ResNet50",
        "YOLOv8 Classification"
    ],

    "Validation Accuracy":[
        "94.12%",
        "97.24%",
        "34.55%",
        "98.31%",
        "98.93%"
    ],

    "Test Accuracy":[
        "93.42%",
        "96.98%",
        "32.92%",
        "97.51%",
        "Validation Evaluated"
    ]
})
