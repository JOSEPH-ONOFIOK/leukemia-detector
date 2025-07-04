import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


model = tf.keras.models.load_model("model/model.keras")


LAST_CONV_LAYER = "top_conv"

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def generate_heatmap(img):
    img_array = preprocess_image(img)

    # Build model to extract gradients
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Apply Grad-CAM formula
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return heatmap
