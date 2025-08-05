'''
Aquí tienes una arquitectura U-Net modificada en Keras/TensorFlow, que incluye una capa LSTM después de cada bloque de codificación (encoder). La entrada es una imagen de tamaño 64x256x1 (alto × ancho × canales), y se adapta la arquitectura para insertar LSTM bidimensionales (conversión a secuencia + LSTM + reconversión a mapa espacial).
'''

import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return x

def lstm_after_conv(x, filters, name_prefix):
    # Cambiar forma de (B, H, W, C) -> (B, W, H*C) para LSTM por columnas
    B, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    x_reshaped = layers.Reshape((W, H * C))(x)
    x_lstm = layers.Bidirectional(layers.LSTM(filters, return_sequences=True), name=f"{name_prefix}_bilstm")(x_reshaped)
    x_restored = layers.Reshape((H, W, 2 * filters))(x_lstm)
    return x_restored

def unet_with_lstm(input_shape=(64, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder 1
    x1 = conv_block(inputs, 32)
    x1_lstm = lstm_after_conv(x1, 32, "enc1")
    p1 = layers.MaxPooling2D((2, 2))(x1_lstm)

    # Encoder 2
    x2 = conv_block(p1, 64)
    x2_lstm = lstm_after_conv(x2, 64, "enc2")
    p2 = layers.MaxPooling2D((2, 2))(x2_lstm)

    # Encoder 3
    x3 = conv_block(p2, 128)
    x3_lstm = lstm_after_conv(x3, 128, "enc3")
    p3 = layers.MaxPooling2D((2, 2))(x3_lstm)

    # Bottleneck
    bn = conv_block(p3, 256)

    # Decoder 3
    u3 = layers.UpSampling2D((2, 2))(bn)
    u3 = layers.Concatenate()([u3, x3_lstm])
    x4 = conv_block(u3, 128)

    # Decoder 2
    u2 = layers.UpSampling2D((2, 2))(x4)
    u2 = layers.Concatenate()([u2, x2_lstm])
    x5 = conv_block(u2, 64)

    # Decoder 1
    u1 = layers.UpSampling2D((2, 2))(x5)
    u1 = layers.Concatenate()([u1, x1_lstm])
    x6 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x6)

    model = Model(inputs, outputs)
    return model

