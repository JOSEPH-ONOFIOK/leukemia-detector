 
from PIL import Image
import numpy as np

def resize_image(image, size=(224, 224)):
    return image.resize(size)

def normalize_image(image):
    arr = np.array(image) / 255.0
    return arr

def preprocess_pipeline(image):
    image = resize_image(image)
    image = normalize_image(image)
    return image
