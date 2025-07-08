# app/main.py

import os
import sys
import streamlit as st
import numpy as np
from PIL import Image

# Add project root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import prediction function and class labels
from model.predict import classify_image, classes

# Set page configuration
st.set_page_config(page_title="Leukemia Detection", layout="wide")

# Sidebar navigation
st.sidebar.title("Leukemia Detection Tool")
page = st.sidebar.radio("Navigate", ["Dashboard", "Upload & Predict", "About"])

# Style for headings
st.markdown("""
    <style>
    .main-title { font-size:2.5rem; font-weight:bold; }
    .sub-title { font-size:1.2rem; color:#555; margin-bottom:20px; }
    </style>
""", unsafe_allow_html=True)

# Dashboard page
if page == "Dashboard":
    st.markdown('<div class="main-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">General insights and model information.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Classes", f"{len(classes)}")
    col2.metric("Model Accuracy", "~92%")
    col3.metric("Model Status", "Ready")

    st.markdown("### Supported Classes")
    st.code(", ".join(classes))

# Upload and predict page
elif page == "Upload & Predict":
    st.markdown('<div class="main-title">Upload & Predict</div>', unsafe_allow_html=True)
    st.markdown("Upload a blood smear image to detect the type of leukemia.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Analyzing image..."):
                prediction, confidence, _ = classify_image(image)

            st.success(f"Prediction: {prediction}")
            st.info(f"Confidence: {confidence:.2f}%")

            treatment_dict = {
                "AML": "Begin induction chemotherapy. Consider bone marrow transplant for high-risk cases.",
                "ALL": "Combination chemotherapy is standard. CAR-T may be used in resistant cases.",
                "CLL": "Often monitored initially. Use targeted therapies if disease progresses.",
                "CML": "Treat with tyrosine kinase inhibitors like imatinib."
            }

            st.subheader("Recommended Follow-Up")
            st.write(treatment_dict.get(prediction, "Consult a hematologist for further evaluation."))

# About page
elif page == "About":
    st.markdown('<div class="main-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    This application uses a trained deep learning model to classify blood smear images into one of four types of leukemia.

    **Supported Classes:**
    - ALL (Acute Lymphoblastic Leukemia)
    - AML (Acute Myeloid Leukemia)
    - CLL (Chronic Lymphocytic Leukemia)
    - CML (Chronic Myeloid Leukemia)

    **Technologies Used:**
    - TensorFlow / Keras
    - Streamlit
    - Python

    **Disclaimer:** This tool is intended for educational and research purposes only and is not a substitute for professional medical diagnosis.
    """)

# Footer
st.markdown("<hr><center><small>&copy; 2025 Group 3 | Leukemia Detector</small></center>", unsafe_allow_html=True)
