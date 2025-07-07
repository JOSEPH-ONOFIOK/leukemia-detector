import streamlit as st
from PIL import Image
import sys
import os

# Make sure Python can access the model/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predict import classify_image
from model.gradcam import generate_heatmap

st.set_page_config(page_title="Leukemia Detector", layout="wide")

st.title("Leukemia Detection System")
st.markdown("Upload a blood smear image and let the AI model classify the **leukemia subtype** and visualize important regions.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        prediction, confidence = classify_image(image)
        heatmap = generate_heatmap(image)

    st.success(f"**Prediction:** `{prediction}` with **{confidence:.2f}% confidence**")
    st.image(heatmap, caption="Grad-CAM Heatmap (Model Attention)", use_column_width=True)
