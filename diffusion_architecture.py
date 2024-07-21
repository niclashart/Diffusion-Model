import math
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
def sinusoidal_embedding(x, embedding_max_frequency: float = 1000.0, embedding_dims: int = 32):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(tf.linspace(tf.math.log(embedding_min_frequency),
                                     tf.math.log(embedding_max_frequency),
                                     embedding_dims // 2,))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        # TODO: Build ResidualBlock consisting of BatchNormalization (center and scale=False), 3x3 Conv2D with swish activation and padding="same", 3x3 Conv2D with padding="same", Add to connect x and residual
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, activation='swish', padding='same')(x)
        x = layers.Conv2D(width, kernel_size=3, padding='same')(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        # TODO: Use AveragePooling2D with a pool_size of 2 for downsampling
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        # TODO: Use UpSampling2D with size 2 and interpolation="bilinear" for upsampling
        x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # Defines the initial Conv2D Layer
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    # Builds the Downsampling Part of the Network
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    # TODO: Implement the for-loop for the Upsampling Part of the Network
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])


    # TODO: Define the final 1x1 Conv2D Layer. Use 'zeros' for kernel_initializer
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer='zeros')(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")
