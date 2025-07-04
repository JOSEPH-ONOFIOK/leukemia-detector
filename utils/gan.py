import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

layers = tf.keras.layers  

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, input_shape=(latent_dim,), use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, 1, 'same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, 5, 2, 'same', activation='tanh', use_bias=False)
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, 2, 'same', input_shape=(28, 28, 3)),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

def train_gan(generator, discriminator, latent_dim=100, epochs=2, batch_size=32):
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = tf.image.resize(x_train / 127.5 - 1, (28, 28))
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_opt = tf.keras.optimizers.Adam(1e-4)
    d_opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            gen_images = generator(noise)
            real_out = discriminator(images)
            fake_out = discriminator(gen_images)
            g_loss = loss_fn(tf.ones_like(fake_out), fake_out)
            d_loss = loss_fn(tf.ones_like(real_out), real_out) + \
                     loss_fn(tf.zeros_like(fake_out), fake_out)
        g_grad = gt.gradient(g_loss, generator.trainable_variables)
        d_grad = dt.gradient(d_loss, discriminator.trainable_variables)
        g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))
        d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))

    for epoch in range(epochs):
        for img_batch in dataset:
            train_step(img_batch)
        noise = tf.random.normal([1, latent_dim])
        img = generator(noise)[0].numpy()
        os.makedirs("data/synthetic", exist_ok=True)
        plt.imsave(f"data/synthetic/generated_{epoch+1}.png", (img + 1) / 2)
        print(f" Epoch {epoch+1}: Image saved to data/synthetic/")


if __name__ == "__main__":
    generator = build_generator(100)
    discriminator = build_discriminator()
    train_gan(generator, discriminator)
