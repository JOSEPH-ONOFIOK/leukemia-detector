import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

# Aliases
EfficientNetB0 = tf.keras.applications.EfficientNetB0
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping
CategoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy

def build_model(num_classes):
    base = EfficientNetB0(include_top=False, input_shape=(224,224,3), weights='imagenet')
    for layer in base.layers:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)

    model.compile(optimizer=Adam(1e-5), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model

def train_model(data_dir='data', output_path='model/model.h5', epochs=30):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224), class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224), class_mode='categorical', subset='validation')

    labels = list(train_gen.class_indices.keys())
    print("Detected classes:", labels)

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    weights = dict(enumerate(weights))

    model = build_model(len(labels))
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, class_weight=weights)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f" Model saved to: {output_path}")

if __name__ == "__main__":
    train_model()
