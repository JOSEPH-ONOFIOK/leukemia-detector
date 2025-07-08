# app/main.py
import os
import sys
import streamlit as st
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.predict import classify_image, classes  # Only import predict

st.set_page_config(page_title="Leukemia Detection", layout="wide")

st.sidebar.title("Leukemia Detection Tool")
page = st.sidebar.radio("Navigate", ["Dashboard", "Upload & Predict", "About"])

st.markdown("""
    <style>
    .main-title { font-size:2.5rem; font-weight:bold; }
    .sub-title { font-size:1.2rem; color:#555; margin-bottom:20px; }
    </style>
""", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown('<div class="main-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">General insights and model info.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Classes", f"{len(classes)}")
    col2.metric("Model Accuracy", "~92%")
    col3.metric("Model Status", "Ready")

    st.markdown("### Classes")
    st.code(", ".join(classes))

elif page == "Upload & Predict":
    st.markdown('<div class="main-title">Upload & Predict</div>', unsafe_allow_html=True)
    st.markdown("Upload a blood smear image to detect leukemia type.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                prediction, confidence, _ = classify_image(image)

            st.success(f"Prediction: {prediction}")
            st.info(f"Confidence: {confidence:.2f}%")

            treatment_dict = {
                "AML": "Start induction chemotherapy. Consider BMT if high-risk.",
                "ALL": "Combination chemo. CAR-T for resistant cases.",
                "CLL": "Monitor. Targeted therapy if progressive.",
                "CML": "Use tyrosine kinase inhibitors like imatinib."
            }
            st.subheader("Follow-up Recommendation")
            st.write(treatment_dict.get(prediction, "Consult a hematologist."))

elif page == "About":
    st.markdown('<div class="main-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    This app classifies blood smear images into 4 leukemia types using a deep learning model.

    **Supported Classes:**
    - ALL
    - AML
    - CLL
    - CML

    **Tech Stack:**
    - TensorFlow/Keras
    - Streamlit

    **Disclaimer:** Not for clinical use.
    """)

st.markdown("<hr><center><small>&copy; 2025 Group 3 | Leukemia Detector</small></center>", unsafe_allow_html=True)
