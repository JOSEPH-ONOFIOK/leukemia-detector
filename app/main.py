import streamlit as st
from PIL import Image
from model.predict import classify_image
from model.gradcam import generate_heatmap

st.set_page_config(page_title="Leukemia Detector", layout="wide")

st.title("Leukemia Detection System")
st.write("Upload a blood smear image and let the AI classify the subtype.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        prediction, confidence = classify_image(image)
        heatmap = generate_heatmap(image)

    st.success(f"Prediction: **{prediction}** ({confidence:.2f}%)")
    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
