import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, downsample=False):
    if downsample:
        x = layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu')(x)
    else:
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def up_block(x, skip, prev_before_lstm, filters):
    # Upsample
    x = layers.UpSampling2D(size=2, interpolation='nearest')(x)
    prev_before_lstm = layers.UpSampling2D(size=2, interpolation='nearest')(prev_before_lstm)
    # Concatenate interpolated LSTM input + upsampled skip connection
    x = layers.Concatenate()([x, prev_before_lstm, skip])
    # Conv to reduce channels and extract features
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def build_decoder(shared_lstm_features, skip_connections, before_lstm, output_channels=1):
    x = shared_lstm_features
    for i, filters in enumerate([256, 128, 64, 32]):
        x = up_block(x, skip_connections[-(i+1)], before_lstm[-(i+1)], filters)
    # Final output conv
    x = layers.Conv2D(output_channels, 3, padding='same')(x)
    return x

# Input
input_img = layers.Input(shape=(192, 192, 1))

# Encoder
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
skips = []  # for skip connections
before_lstm = []  # maps before BiLSTM for concat later

for filters in [64, 128, 256, 512]:
    x = conv_block(x, filters, downsample=True)
    skips.append(x)
    before_lstm.append(x)  # to use later for upsampling

# Latent space: 12x12x512 → 12x12x32
x = layers.Conv2D(32, 1, activation='relu')(x)
latent_features = x  # shape: (None, 12, 12, 32)

# Flatten spatial dimensions for LSTM
b, h, w, c = tf.unstack(tf.shape(latent_features))
x_reshaped = layers.Reshape((h*w, c))(latent_features)
x_bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x_reshaped)

# Reshape back to spatial format: 12x12x256
x_bilstm = layers.Reshape((h, w, 256))(x_bilstm)

# Compartir esta salida para 3 decodificadores
outputs = []
for name in ['u', 'v', 'P']:
    decoder_output = build_decoder(x_bilstm, skips, before_lstm)
    outputs.append(decoder_output)

# Modelo final
model = Model(inputs=input_img, outputs=outputs, name="GeoLSTM_Unet3Head")
model.summary()

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(model, dataset, num_batches=1):
    for batch in dataset.take(num_batches):
        x_batch, (y_u_true, y_v_true, y_p_true) = batch
        y_pred = model.predict(x_batch)

        for i in range(len(x_batch)):
            fig, axs = plt.subplots(3, 4, figsize=(16, 10))
            titles = ['u', 'v', 'P']
            y_true = [y_u_true, y_v_true, y_p_true]
            y_hat = y_pred

            for j in range(3):  # Para u, v, P
                # Entrada (escala de grises)
                im0 = axs[j, 0].imshow(x_batch[i, :, :, 0], cmap='gray')
                axs[j, 0].set_title('Input')
                axs[j, 0].axis('off')

                # Real
                im1 = axs[j, 1].imshow(y_true[j][i, :, :, 0], cmap='viridis')
                axs[j, 1].set_title(f'Real {titles[j]}')
                axs[j, 1].axis('off')
                plt.colorbar(im1, ax=axs[j, 1], fraction=0.046)

                # Predicción
                im2 = axs[j, 2].imshow(y_hat[j][i, :, :, 0], cmap='viridis')
                axs[j, 2].set_title(f'Predicted {titles[j]}')
                axs[j, 2].axis('off')
                plt.colorbar(im2, ax=axs[j, 2], fraction=0.046)

                # Error absoluto
                error = np.abs(y_true[j][i, :, :, 0] - y_hat[j][i, :, :, 0])
                im3 = axs[j, 3].imshow(error, cmap='hot')
                axs[j, 3].set_title(f'Abs Error {titles[j]}')
                axs[j, 3].axis('off')
                plt.colorbar(im3, ax=axs[j, 3], fraction=0.046)

            plt.tight_layout()
            plt.show()


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_predictions(model, dataset, num_batches=1):
    for batch in dataset.take(num_batches):
        x_batch, (y_u_true, y_v_true, y_p_true) = batch
        y_pred = model.predict(x_batch)

        for i in range(len(x_batch)):
            fig, axs = plt.subplots(3, 4, figsize=(18, 10))
            titles = ['u', 'v', 'P']
            y_true = [y_u_true, y_v_true, y_p_true]
            y_hat = y_pred

            for j in range(3):  # Para u, v, P
                input_img = x_batch[i, :, :, 0]
                true_img = y_true[j][i, :, :, 0]
                pred_img = y_hat[j][i, :, :, 0]
                error_img = np.abs(true_img - pred_img)

                # Métricas
                mse = np.mean((true_img - pred_img) ** 2)
                mae = np.mean(error_img)
                rmse = np.sqrt(mse)

                # Entrada
                im0 = axs[j, 0].imshow(input_img, cmap='gray')
                axs[j, 0].set_title('Input')
                axs[j, 0].axis('off')

                # Real
                im1 = axs[j, 1].imshow(true_img, cmap='viridis')
                axs[j, 1].set_title(f'Real {titles[j]}')
                axs[j, 1].axis('off')
                plt.colorbar(im1, ax=axs[j, 1], fraction=0.046)

                # Predicho
                im2 = axs[j, 2].imshow(pred_img, cmap='viridis')
                axs[j, 2].set_title(f'Predicted {titles[j]}')
                axs[j, 2].axis('off')
                plt.colorbar(im2, ax=axs[j, 2], fraction=0.046)

                # Error absoluto con métricas
                im3 = axs[j, 3].imshow(error_img, cmap='hot')
                axs[j, 3].set_title(
                    f'Abs Error {titles[j]}\nMSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}'
                )
                axs[j, 3].axis('off')
                plt.colorbar(im3, ax=axs[j, 3], fraction=0.046)

            plt.suptitle(f'Predicciones vs Reales - Ejemplo {i}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

