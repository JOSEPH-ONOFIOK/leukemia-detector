import streamlit as st
from PIL import Image
import sys
import os
import io
import base64

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)


from predict import classify_image, model, classes
from gradcam import generate_heatmap

# Streamlit config
st.set_page_config(page_title="Leukemia Detector", layout="wide")
st.title(" Leukemia Detection System")
st.write("Upload a blood smear image and let the AI classify the subtype of leukemia.")


uploaded_image = st.file_uploader(" Upload Image", type=["jpg", "jpeg", "png"])

# image anaylsis
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        # Classification of images
        prediction, confidence = classify_image(image)

        # Grad-CAM heatmap
        try:
            heatmap, overlay = generate_heatmap(image)
        except Exception as e:
            heatmap = None
            overlay = None
            st.warning("grad-cam doesnt exist for image")

    
    st.success(f" Prediction: **{prediction}** ({confidence:.2f}%)")

    
    if heatmap is not None:
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

        
        buffered = io.BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode()

        href = f'<a href="data:image/png;base64,{b64}" download="gradcam_overlay.png"> Download Heatmap</a>'
        st.markdown(href, unsafe_allow_html=True)
