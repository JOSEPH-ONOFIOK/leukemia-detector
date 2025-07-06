import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.losses import CategoricalCrossentropy

# Match the custom loss config from training
custom_loss = CategoricalCrossentropy(label_smoothing=0.1)

# Load model with matching custom loss
model = tf.keras.models.load_model(
    "model/model.keras",
    custom_objects={"CategoricalCrossentropy": custom_loss}
)

# Class labels
classes = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def classify_image(image):
    input = preprocess_image(image)
    preds = model.predict(input)[0]
    idx = np.argmax(preds)
    return classes[idx], preds[idx] * 100
