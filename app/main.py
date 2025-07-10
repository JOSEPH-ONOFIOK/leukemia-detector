import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import models
from model.predict import classify_image, classes
from model.evaluate import get_classification_report

# Page config
st.set_page_config(page_title="Leukemia Detection Dashboard", layout="wide")

# Global CSS styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: #f5f8fa;
            color: #000000;
        }

        section[data-testid="stSidebar"] {
            background-color: #003366;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] .stMarkdown {
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
page = st.sidebar.radio("Navigate", ["Dashboard", "Upload & Predict", "Model Evaluation", "About"])

# Dashboard page
if page == "Dashboard":
    st.markdown("""
        <div class="main-title">Dashboard Overview</div>
        <div class="sub-title">Explore model insights and learn how to use the platform effectively.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='margin-top:30px; margin-bottom:20px;'>
            <h4 style='color:#003366;'>Model Quick Stats</h4>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Classes", value=f"{len(classes)} types")
    with col2:
        st.metric(label="Model Accuracy", value="~92%")
    with col3:
        st.metric(label="Model Status", value="Ready")

    st.markdown("""
        <div style='margin-top:30px; margin-bottom:10px;'>
            <h4 style='color:#003366;'>Supported Leukemia Classes</h4>
        </div>
    """, unsafe_allow_html=True)
    st.code(", ".join(classes))

    st.markdown("""
        <div style='margin-top:30px; margin-bottom:10px;'>
            <h4 style='color:#003366;'>How to Use This Application</h4>
            <ul style='font-size: 16px; line-height: 1.8;'>
                <li>Go to the <strong>Upload & Predict</strong> section.</li>
                <li>Enter <strong>patient details</strong>: name, age, gender, and blood type.</li>
                <li>Upload a <strong>clear blood smear image</strong> (JPG/PNG).</li>
                <li>Wait for diagnosis with:
                    <ul>
                        <li>Predicted Leukemia Type</li>
                        <li>Confidence Score</li>
                        <li>Recommended Treatment</li>
                    </ul>
                </li>
                <li>Optionally <strong>download the report as a PDF</strong>.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Upload & Predict Page
elif page == "Upload & Predict":
    st.markdown('<div class="main-title">Leukemia Cell Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a blood smear image and enter patient details for diagnosis.</div>', unsafe_allow_html=True)

    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120)
    with col2:
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
        blood_type = st.selectbox("Blood Type", ["Select", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

    st.markdown("Upload Blood Smear Image")

    if name and age and gender != "Select" and blood_type != "Select":
        uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])
            image = Image.open(uploaded_file).convert("RGB")

            with col1:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                with st.spinner("Running inference..."):
                    prediction, confidence, _ = classify_image(image)

                st.markdown(f"<div style='font-size: 20px; color: black;'><strong>Predicted Type:</strong> {prediction}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 18px; color: black;'><strong>Confidence Score:</strong> {confidence:.2f}%</div>", unsafe_allow_html=True)

                treatment_dict = {
                    "AML": "Begin induction chemotherapy. Consider bone marrow transplant if risk is high.",
                    "ALL": "Combination chemotherapy, possibly CAR-T therapy for refractory cases.",
                    "CLL": "Often managed with watchful waiting. Use targeted therapy if progressive.",
                    "CML": "Use tyrosine kinase inhibitors such as imatinib."
                }

                recommendation = treatment_dict.get(prediction, "Consult a hematologist for a full clinical evaluation.")
                st.subheader("Recommended Follow-up")
                st.write(f"**Suggested Action:** {recommendation}")

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
                    c.drawString(50, 630, "Recommended Follow-up:")
                    c.drawString(70, 610, recommendation)
                    c.showPage()
                    c.save()
                    buffer.seek(0)
                    return buffer

                if st.button("Download Diagnosis Report as PDF"):
                    pdf = create_pdf()
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf,
                        file_name=f"{name.replace(' ', '_')}_leukemia_report.pdf",
                        mime="application/pdf"
                    )
    else:
      st.markdown("<div style='color: red; font-weight: bold;'>Please complete all patient fields before uploading an image.</div>", unsafe_allow_html=True)


# Model Evaluation Page
elif page == "Model Evaluation":
    st.markdown('<div class="main-title">Model Evaluation Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review classification performance metrics on the validation dataset.</div>', unsafe_allow_html=True)
    report = get_classification_report()
    st.text_area("Classification Report", report, height=400)

# About Page
elif page == "About":
    st.markdown("""
        <div class="main-title">About This Application</div>
        <div class="sub-title">Get to know the technology and purpose behind this project.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='padding: 20px; background-color: #ffffff; border-radius: 10px; border: 1px solid #ccc; margin-top: 20px; font-size: 16px;'>
            <p>This web-based tool leverages <strong>deep learning</strong> to detect types of leukemia from blood smear images. It assists users with preliminary classifications based on trained models.</p>

            <h5 style='color:#003366;'>Supported Classes:</h5>
            <ul>
                <li><strong>ALL</strong> – Acute Lymphoblastic Leukemia</li>
                <li><strong>AML</strong> – Acute Myeloid Leukemia</li>
                <li><strong>CLL</strong> – Chronic Lymphocytic Leukemia</li>
                <li><strong>CML</strong> – Chronic Myeloid Leukemia</li>
            </ul>

            <h5 style='color:#003366;'>Technologies Used:</h5>
            <ul>
                <li>TensorFlow / Keras for Convolutional Neural Network modeling</li>
                <li>Streamlit for interactive front-end dashboard</li>
                <li>ReportLab for generating PDF medical summaries</li>
            </ul>

            <h5 style='color:#d9534f;'>Disclaimer:</h5>
            <p>This tool is intended <strong>only for educational and research purposes</strong>. It is <u>not a substitute</u> for professional medical diagnosis or treatment.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="margin-top: 40px; border: none; height: 1px; background-color: #ccc;" />
    <center style="color: #888;">
        <small>&copy; 2025 <strong>Leukemia Detection and Classification Tool</strong> | Built by Group 3 (Olaniyan)</small>
    </center>
""", unsafe_allow_html=True)
