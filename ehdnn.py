'''
Aquí tienes una implementación en TensorFlow/Keras de la arquitectura eHDNN (Enhanced Hybrid Deep Neural Network) siguiendo las especificaciones que diste: una CNN profunda, una capa ConvLSTM y una DeCNN con activación LeakyReLU. La estructura general está organizada como un Model de Keras modular:
'''



import tensorflow as tf
from tensorflow.keras import layers, models

def build_eHDNN(input_shape):
    inputs = tf.keras.Input(shape=input_shape)  # e.g., (T, H, W, C)

    # ▪ Capa Convolucional (CNN) Profunda - 7 capas 2D
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(inputs)
    for _ in range(6):
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)

    # ▪ Transposición para ConvLSTM (cambiar a formato [batch, time, height, width, channels])
    # Ya está en formato correcto si `inputs` tiene esa forma.

    # ▪ Capa de Memoria (ConvLSTM) - 4 capas
    for _ in range(3):
        x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, activation='tanh')(x)
    x = layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False, activation='tanh')(x)

    # ▪ Capa Deconvolucional (DeCNN) - Reproyección al espacio original con LeakyReLU
    for _ in range(3):
        x = layers.Conv2DTranspose(64, (3, 3), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    
    # ▪ Última capa para producir salida del campo de flujo (e.g., 1 o más canales)
    output_channels = 3  # u, v, P, por ejemplo
    outputs = layers.Conv2DTranspose(output_channels, (3, 3), padding='same', activation='linear')(x)

    return tf.keras.Model(inputs, outputs, name='eHDNN')

