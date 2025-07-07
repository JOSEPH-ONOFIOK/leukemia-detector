import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.keras")
model = tf.keras.models.load_model(model_path)

# Define class labels
classes = ['ALL', 'AML', 'CLL', 'CML']  # Adjust based on your training

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def classify_image(image):
    input_tensor = preprocess_image(image)
    preds = model.predict(input_tensor)[0]  # shape: (num_classes,)
    idx = np.argmax(preds)
    confidence = preds[idx] * 100
    return classes[idx], confidence, preds
