import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from predict import model, preprocess_image

# Define the last conv layer name manually based on your model
LAST_CONV_LAYER = "conv2d_7"

def generate_heatmap(pil_img):
    img_array = preprocess_image(pil_img)

    # a model that returns last conv output + predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Gradient of the class with respect to conv output
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert original image to OpenCV format
    img_cv = np.array(pil_img.resize((224, 224)))
    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    elif img_cv.shape[-1] == 1:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_colored, overlay
