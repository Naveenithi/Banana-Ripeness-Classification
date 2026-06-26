
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import tempfile
import time
import os

st.set_page_config(
    page_title="🍌 Banana Ripeness Classification",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("best.pt not found in the project folder.")
        st.stop()
    return YOLO(model_path)

model = load_model()
CLASS_NAMES = model.names

comparison = pd.DataFrame({
    "Model":["Custom CNN","MobileNetV2","EfficientNetB0","ResNet50","YOLOv8"],
    "Validation Accuracy":["94.12%","97.24%","34.55%","98.31%","98.93%"],
    "Test Accuracy":["93.42%","96.98%","32.92%","97.51%","Validation Evaluated"]
})

st.sidebar.title("🍌 Banana Ripeness")
st.sidebar.success("YOLOv8 Classification")
st.sidebar.metric("Validation Accuracy","98.93%")
st.sidebar.write("**Dataset**")
st.sidebar.write("13,478 Images")
st.sidebar.write("4 Classes")
st.sidebar.write("**Classes**")
for c in CLASS_NAMES.values():
    st.sidebar.write(f"• {c.title()}")

st.title("🍌 Banana Ripeness Classification")
st.caption("Deep Learning Based Fruit Quality Assessment")

uploaded = st.file_uploader("Upload a banana image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    c1,c2 = st.columns([1,1])

    with c1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as f:
        image.save(f.name)
        start=time.time()
        results=model.predict(f.name, verbose=False)
        elapsed=time.time()-start

    probs=results[0].probs.data.cpu().numpy()
    pred=int(np.argmax(probs))
    conf=float(probs[pred])

    with c2:
        st.subheader("Prediction")
        st.metric("Class", CLASS_NAMES[pred].title())
        st.metric("Confidence", f"{conf*100:.2f}%")
        st.metric("Inference Time", f"{elapsed:.2f} sec")

        chart=pd.DataFrame(
            {"Confidence (%)":[round(float(x*100),2) for x in probs]},
            index=[CLASS_NAMES[i].title() for i in range(len(probs))]
        )
        st.subheader("Prediction Confidence")
        st.bar_chart(chart)

st.markdown("---")
st.subheader("Model Comparison")
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("About")
st.write("""
This application predicts banana ripeness using a trained YOLOv8 Classification model.

**Classes**
- Overripe
- Ripe
- Rotten
- Unripe
""")

st.markdown("---")
st.caption("Developed by Naveen Kumar")
