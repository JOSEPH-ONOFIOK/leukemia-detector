import os
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

# Path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
VAL_DIR = os.path.join(BASE_DIR, '..', 'data', 'val')

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the validation dataset instead of test dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = val_ds.class_names

# Predict
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds_labels = np.argmax(preds, axis=1)
    y_pred.extend(preds_labels)
    y_true.extend(labels.numpy())

report = classification_report(y_true, y_pred, target_names=class_names)

def get_classification_report():
    return report
