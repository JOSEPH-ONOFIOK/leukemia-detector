import os
import sys
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
from streamlit_lottie import st_lottie
import requests

# Setup project path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

#  Import model utilities
from model.predict import classify_image, classes
from model.load_model import model
from model.saliency import get_img_array, compute_saliency_map, overlay_saliency

# Streamlit UI Config
st.set_page_config(page_title="Leukemia Detection Dashboard", layout="wide")

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Global CSS Styling 
st.markdown("""
    <style>
        body, .stApp {
            background-color: #f5f8fa;
            color: #000000;
        }

        section[data-testid="stSidebar"] {
            background-color: #003366;
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #0d6efd;
        }

        .sub-title {
            font-size: 1.2rem;
            color: #5c7c8a;
            margin-bottom: 20px;
        }

        .stMetric label, .stMetric div {
            color: #000000 !important;
        }

        label {
            color: #000000 !important;
        }

        .stButton>button {
            background-color: #0a9396;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
        }

        .stButton>button:hover {
            background-color: #007f85;
        }

        .image-box {
            border: 1px solid #ccc;
            padding: 10px;
            border-radius: 8px;
            background-color: #ffffff;
        }

        textarea {
            background-color: #ffffff !important;
            color: black !important;
        }

        hr {
            border: none;
            height: 1px;
            background-color: #ccc;
        }

        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Leukemia Detection Tool")
page = st.sidebar.radio("Navigate", ["Home", "Upload & Predict", "About"])

# Home
if page == "Home":
    st.title("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Classes", f"{len(classes)} types")
    col3.metric("Model Status", "Ready")

    st.subheader("Supported Leukemia Classes")
    st.code(", ".join(classes))

    st.subheader("How to Use This Application")
    st.markdown("""
        1. Go to **Upload & Classify**
        2. Enter **patient details**
        3. Upload a **blood smear image**
        4. Get **diagnosis**, **confidence**, **treatment** & **PDF report**
    """)

# Upload & Predict Page 
elif page == "Upload & Predict":
    st.title("Leukemia Cell Classifier")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120)
    with col2:
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        blood_type = st.selectbox("Blood Type", ["Select", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    if name and age and gender != "Select" and blood_type != "Select":
        uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            with st.container():
                with st.spinner("Preparing diagnosis..."):
                    lottie = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json")
                    if lottie:
                        st_lottie(lottie, speed=1, height=150, key="loading_lottie")

                    img_array = get_img_array(image)
                    preds = model.predict(img_array)[0]
                    pred_class_index = np.argmax(preds)
                    prediction = classes[pred_class_index]
                    confidence = preds[pred_class_index] * 100

            st.markdown(f"""
                <div style="padding:10px; border:2px solid #ff4b4b; border-radius:8px; background-color:#fff3f3;">
                    <span style="font-size: 18px; font-weight: bold; color: #d60000;">Predicted Type: {prediction}</span><br>
                    <span>Confidence Score: {confidence:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)

            treatment_dict = {
                "AML": "Begin induction chemotherapy. Consider bone marrow transplant.",
                "ALL": "Combination chemotherapy, possibly CAR-T therapy.",
                "CLL": "Watchful waiting or targeted therapy.",
                "CML": "Tyrosine kinase inhibitors like imatinib."
            }
            recommendation = treatment_dict.get(prediction, "Consult a hematologist.")
            st.subheader("Recommended Follow-up")
            st.write(recommendation)

            col_img, col_sal = st.columns(2)
            with col_img:
                st.markdown("**Uploaded Image**")
                st.image(image, use_column_width=True)

            if st.checkbox(" **:red[Show Saliency Map Explanation]**"):
                try:
                    saliency_map = compute_saliency_map(model, img_array, class_index=pred_class_index)
                    saliency_img = overlay_saliency(image, saliency_map)
                    with col_sal:
                        st.markdown("**Saliency Map**")
                        st.image(saliency_img, use_column_width=True)
                except Exception as e:
                    st.error(f"Saliency map generation failed: {e}")

            def create_pdf():
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                c.setFont("Helvetica", 12)
                c.drawString(50, 750, f"Patient Name: {name}")
                c.drawString(50, 730, f"Age: {age}")
                c.drawString(50, 710, f"Gender: {gender}")
                c.drawString(50, 690, f"Blood Type: {blood_type}")
                c.drawString(50, 670, f"Predicted Leukemia Type: {prediction}")
                c.drawString(50, 650, f"Confidence Score: {confidence:.2f}%")
                c.drawString(50, 630, f"Recommended Follow-up: {recommendation}")
                c.showPage()
                c.save()
                buffer.seek(0)
                return buffer

            if st.button("Download Diagnosis Report as PDF"):
                pdf = create_pdf()
                st.download_button("Click to Download PDF", data=pdf, file_name=f"{name.replace(' ', '_')}_leukemia_report.pdf", mime="application/pdf")
    else:
        st.warning("Please complete all patient fields before uploading an image.")

# About Page
elif page == "About":
    st.title("About This Application")
    st.write("""
        This tool uses deep learning to detect leukemia from blood smear images.

        **Supported Classes:** ALL, AML, CLL, CML  
        **Technologies:** TensorFlow, Streamlit, Saliency Maps, ReportLab, Lottie  
        **Disclaimer:** For educational and research purposes only. Not a substitute for medical advice.
        **Developed by:** GROUP 3, Dr.(Mrs) Olaninyan.
    """)
