from tensorflow.keras.models import load_model
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")


model = load_model(MODEL_PATH)
