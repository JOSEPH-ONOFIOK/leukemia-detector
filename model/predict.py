import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.losses import CategoricalCrossentropy

# Load the trained model with custom_objects for CategoricalCrossentropy
model = tf.keras.models.load_model(
    "model/model.keras",
    custom_objects={"CategoricalCrossentropy": CategoricalCrossentropy}
)

# Define class labels
classes = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

# Preprocess the input image
def preprocess_image(img):
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# Make prediction
def classify_image(image):
    input = preprocess_image(image)
    preds = model.predict(input)[0]
    idx = np.argmax(preds)
    return classes[idx], preds[idx] * 100
