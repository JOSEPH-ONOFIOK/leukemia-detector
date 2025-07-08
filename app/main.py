import os
import sys
import streamlit as st
import numpy as np
from PIL import Image

# Add the root directory to the system path so we can import from the model directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import model prediction and evaluation functions
from model.predict import classify_image, classes
from model.evaluate import get_classification_report

# Streamlit page configuration
st.set_page_config(page_title="Leukemia Detection Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Leukemia Detection Tool")
page = st.sidebar.radio("Navigate", ["Dashboard", "Upload & Predict", "Model Evaluation", "About"])

# Custom styles for headers
st.markdown("""
    <style>
    .main-title { font-size:2.5rem; font-weight:bold; }
    .sub-title { font-size:1.2rem; color:#555; margin-bottom:20px; }
    </style>
""", unsafe_allow_html=True)

# Dashboard Overview
if page == "Dashboard":
    st.markdown('<div class="main-title">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review general information about the model and its classes.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Classes", value=f"{len(classes)} types")
    with col2:
        st.metric(label="Model Accuracy", value="~92%")
    with col3:
        st.metric(label="Model Status", value="Ready")

    st.markdown("### Model Class Labels")
    st.code(", ".join(classes))

# Upload and Predict Page
elif page == "Upload & Predict":
    st.markdown('<div class="main-title">Leukemia Cell Classifier</div>', unsafe_allow_html=True)
    st.markdown("Upload a blood smear image to receive classification and treatment recommendations.")

    uploaded_file = st.file_uploader("Choose a blood smear image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify image
        with col2:
            with st.spinner("Running inference..."):
                prediction, confidence, _ = classify_image(image)

            st.success(f"Predicted Type: `{prediction}`")
            st.info(f"Confidence Score: `{confidence:.2f}%`")

            st.subheader("Recommended Follow-up")
            treatment_dict = {
                "AML": "Begin induction chemotherapy. Consider bone marrow transplant if risk is high.",
                "ALL": "Combination chemotherapy, possibly CAR-T therapy for refractory cases.",
                "CLL": "Often managed with watchful waiting. Use targeted therapy if progressive.",
                "CML": "Use tyrosine kinase inhibitors such as imatinib."
            }

            recommendation = treatment_dict.get(prediction, "Consult a hematologist for a full clinical evaluation.")
            st.write(f"Suggested Action: {recommendation}")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.markdown('<div class="main-title">Model Evaluation Report</div>', unsafe_allow_html=True)
    st.markdown("Review classification performance metrics on the validation dataset.")
    report = get_classification_report()
    st.text_area("Classification Report", report, height=400)

# About Page
elif page == "About":
    st.markdown('<div class="main-title">About This Application</div>', unsafe_allow_html=True)
    st.markdown("""
    This web application is designed to classify types of leukemia from blood smear images using a deep learning model.

    **Supported Classes**:
    - Acute Lymphoblastic Leukemia (ALL)
    - Acute Myeloid Leukemia (AML)
    - Chronic Lymphocytic Leukemia (CLL)
    - Chronic Myeloid Leukemia (CML)

    **Technologies Used**:
    - TensorFlow / Keras for CNN-based image classification
    - Streamlit for building the user interface

    **Disclaimer**:
    This tool is intended for educational and research purposes only. It is not a replacement for professional medical diagnosis.
    """)

# Footer
st.markdown("""
    <hr style="margin-top:40px;">
    <center><small>&copy; 2025 Leukemia Detection Tool | Developed by Group 3 olaniyan</small></center>
""", unsafe_allow_html=True)
