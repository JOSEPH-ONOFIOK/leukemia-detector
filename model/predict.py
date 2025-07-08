import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
model = load_model(MODEL_PATH)

# Force the model to build by calling it with dummy input
model(tf.zeros((1, 224, 224, 3)))

# Define class labels
classes = ['ALL', 'AML', 'CLL', 'CML']

# Image size expected by the model
IMG_SIZE = 224

# Preprocess the image for prediction
def preprocess_image(img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# Make a prediction using the model
def classify_image(img_pil):
    img_array = preprocess_image(img_pil)
    preds = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence, img_array
