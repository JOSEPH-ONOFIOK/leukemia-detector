import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

EfficientNetB0 = tf.keras.applications.EfficientNetB0
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping

def build_model(num_classes):
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir='data', output_path='model/model.h5', epochs=50):
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise FileNotFoundError(f"No data found in '{data_dir}'.")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_labels = list(train_generator.class_indices.keys())
    print(f"\nClasses detected: {class_labels}")

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(len(class_labels))

    callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

    print("Starting training...\n")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")

if __name__ == "__main__":
    train_model()
