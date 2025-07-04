import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


def load_validation_generator(data_dir='data', image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False  
    )

    return val_generator


def evaluate_model(model_path='model/model.keras', data_dir='data', save_plot=False):
    
    model = tf.keras.models.load_model(model_path)

    
    val_generator = load_validation_generator(data_dir)
    class_labels = list(val_generator.class_indices.keys())

    
    Y_pred = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes

    
    print(" Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_plot:
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/confusion_matrix.png")
        print(" Confusion matrix saved to: outputs/confusion_matrix.png")

    plt.show()


if __name__ == "__main__":
    evaluate_model()
