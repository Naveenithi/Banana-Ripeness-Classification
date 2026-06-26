import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="🍌 Banana Ripeness Classification",
    page_icon="🍌",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton>button {
    width:100%;
    border-radius:10px;
}

.prediction-box{
    background-color:#F7F7F7;
    padding:20px;
    border-radius:15px;
    border:1px solid #E6E6E6;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    if not os.path.exists("best.pt"):
        st.error("❌ best.pt not found.")
        st.stop()

    return YOLO("best.pt")

model = load_model()

CLASS_NAMES = model.names

# ===============================
# SIDEBAR
# ===============================

st.sidebar.image(
    "https://img.icons8.com/color/96/banana.png",
    width=80
)

st.sidebar.title("Banana Ripeness")

st.sidebar.markdown("---")

st.sidebar.success("YOLOv8 Classification")

st.sidebar.metric(
    label="Validation Accuracy",
    value="98.93%"
)

st.sidebar.markdown("### Dataset")

st.sidebar.write("📸 13,478 Images")

st.sidebar.write("🏷️ 4 Classes")

st.sidebar.markdown("---")

st.sidebar.markdown("### Classes")

st.sidebar.write("🍌 Overripe")

st.sidebar.write("🍌 Ripe")

st.sidebar.write("🍌 Rotten")

st.sidebar.write("🍌 Unripe")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Developed by

**Naveen Kumar**
"""
)

---

### Classes

🍌 Overripe

🍌 Ripe

🍌 Rotten

🍌 Unripe

---

### Framework

- Ultralytics
- PyTorch
- Streamlit
""")

# ===============================
# HERO SECTION
# ===============================

st.markdown("""
<h1 style='text-align: center; color: #2E8B57;'>
🍌 Banana Ripeness Classification
</h1>

<h4 style='text-align: center; color: gray;'>
Deep Learning Based Fruit Quality Assessment
</h4>

<p style='text-align: center; font-size:18px;'>
🤖 <b>Model:</b> YOLOv8 Classification &nbsp;&nbsp;&nbsp;
🎯 <b>Validation Accuracy:</b> 98.93%
</p>

""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Banana Image",
    type=["jpg","jpeg","png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:

        image.save(temp.name)

        start = time.time()

        with st.spinner("Predicting..."):

            results = model.predict(
                source=temp.name,
                verbose=False
            )

        end = time.time()

    result = results[0]

    probs = result.probs.data.cpu().numpy()

    predicted = int(np.argmax(probs))

    confidence = float(probs[predicted])

    inference = end-start

    with col2:

        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

        st.success("Prediction Completed")

        st.metric(
            "Prediction",
            CLASS_NAMES[predicted].title()+" 🍌"
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.metric(
            "Inference Time",
            f"{inference:.2f} sec"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.subheader("Prediction Confidence")

        for idx, prob in enumerate(probs):

            st.write(f"**{CLASS_NAMES[idx].title()}**")

            st.progress(float(prob))

            st.caption(f"{prob*100:.2f}%")

# -----------------------------
# ABOUT
# -----------------------------
st.markdown("---")

st.subheader("📖 About Project")

st.write("""
This project predicts the ripeness stage of bananas using a **YOLOv8 Classification**
deep learning model trained on a banana image dataset.

### Ripeness Classes

- 🍌 Overripe
- 🍌 Ripe
- 🍌 Rotten
- 🍌 Unripe

### Model

YOLOv8 Classification

Validation Accuracy: **98.93%**

### Technologies Used

- Python
- Streamlit
- Ultralytics YOLOv8
- PyTorch
- NumPy
- Pillow
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
"""
<div class='footer'>
Developed by <b>Naveen Kumar</b> |
Banana Ripeness Classification using YOLOv8
</div>
""",
unsafe_allow_html=True
)
