import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import models
from PIL import Image
import matplotlib.cm as cm
import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
model = load_model(MODEL_PATH)

# Force model to build by calling it once with dummy data
model(tf.zeros((1, 224, 224, 3)))

# Define class names
classes = ['ALL', 'AML', 'CLL', 'CML']

# Image size
IMG_SIZE = 224

# Preprocess image
def preprocess_image(img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# Classify image
def classify_image(img_pil):
    img_array = preprocess_image(img_pil)
    preds = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence, img_array

# Grad-CAM visualization
def get_gradcam_overlay(img_pil):
    img_array = preprocess_image(img_pil)

    # automatically find the last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
        elif hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    last_conv_layer = sublayer
                    break
            if last_conv_layer:
                break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the model.")

    # Create model that maps input to the last conv layer and output
    grad_model = models.Model(
        [model.input],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize and convert to numpy
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)

    # Colorize heatmap
    colormap = cv2.COLORMAP_JET
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)

    # Overlay on original image
    original_img = img_pil.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original_img.convert("RGB"))
    overlayed_img = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)

    return Image.fromarray(overlayed_img)
