Leukemia Detection System using Deep Learning

This project is an AI-powered leukemia detection tool that classifies blood smear images into five leukemia types (ALL, AML, CLL, CML) and Healthy samples using a fine-tuned EfficientNet model with Grad-CAM explainability.



Features

- Classifies blood smear images into:
  - ALL – Acute Lymphoblastic Leukemia
  - AML – Acute Myeloid Leukemia
  - CLL – Chronic Lymphocytic Leukemia
  - CML– Chronic Myeloid Leukemia
  - Healthy – No cancerous cells detected
  -  Grad-CAM heatmap for interpretability
  - Model evaluation with classification report & confusion matrix
  -  GAN support for synthetic image generation (optional)
  - Streamlit interface for ease of use
  - Basic test suite for model predictions


Tech Stack

- TensorFlow / Keras – Model training & inference
- EfficientNetB0 – Base architecture
- Streamlit – Frontend interface
- OpenCV & Matplotlib** – Visualization
- scikit-learn – Evaluation tools
- Seaborn– Confusion matrix heatmap




Installation

git clone  git remote add origin https://github.com/JOSEPH-ONOFIOK/leukemia-detector.git
cd leukemia-detector
pip install -r requirements.txt



