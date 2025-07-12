import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

def get_img_array(image, size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.convert_to_tensor(img_array[np.newaxis, ...])  # Shape: (1, 224, 224, 3)
    return img_array

def compute_saliency_map(model, img_array, class_index=None):
    img_array = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        preds = model(img_array, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, img_array)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]  # Shape: (224, 224)
    saliency = tf.maximum(saliency, 0) / (tf.reduce_max(saliency) + 1e-10)
    return saliency.numpy()

def overlay_saliency(original_img, saliency_map, alpha=0.5):
    image = np.array(original_img.resize((224, 224)))
    heatmap = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(superimposed)
