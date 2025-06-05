import tensorflow as tf

class DownBlock(tf.keras.Model):
    def __init__(self, conv_kernels, lstm_params_list, stride, data_format='channels_last', name='down_block'):
        """
        Inicializa el DownBlock.
        Args:
            conv_kernels: Lista de diccionarios para las capas Conv2D.
                          Cada item: {'filters': int, 'kernel_size': int o tupla, 'padding': 'same'}
            lstm_params_list: Lista de diccionarios para las capas LSTM.
                              Cada item: {'units': int}.
            stride: Stride para la primera capa convolucional (para downsampling).
            data_format: 'channels_first' o 'channels_last'.
        """
        super(DownBlock, self).__init__(name=name)
        self.conv_layers = []
        self.bn_layers = []
        self.leaky_relu_layers = []
        self.lstm_layers = []
        self.data_format = data_format
        self.bn_axis = -1 if data_format == 'channels_last' else 1

        # Capas Convolucionales
        for i, conv_params in enumerate(conv_kernels):
            s = stride if i == 0 else conv_params.get('strides', 1) # Aplicar stride principal a la primera conv
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=conv_params['filters'],
                    kernel_size=conv_params['kernel_size'],
                    strides=s,
                    padding=conv_params.get('padding', 'same'),
                    data_format=data_format,
                    name=f"{name}_conv2d_{i}"
                )
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(axis=self.bn_axis, name=f"{name}_bn_{i}")
            )
            self.leaky_relu_layers.append(
                tf.keras.layers.LeakyReLU(alpha=0.2, name=f"{name}_leakyrelu_{i}")
            )

        # Capas LSTM
        self.original_spatial_dims_for_lstm = None 

        for i, lstm_params in enumerate(lstm_params_list):
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    units=lstm_params['units'],
                    return_sequences=True, 
                    stateful=True, 
                    name=f"{name}_lstm_{i}"
                )
            )
        
        self.current_lstm_states = [None] * len(self.lstm_layers)
        self.pending_initial_lstm_states = None


    def call(self, inputs, training=False):
        """
        Pase hacia adelante para el DownBlock.
        Args:
            inputs: Tensor de entrada.
            training: Booleano, si el modelo está en modo de entrenamiento.
        Returns:
            Tupla: (tensor_procesado_lstm, skip_connection_conv)
        """
        x = inputs
        # Aplicar capas Conv2D secuencialmente
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x, training=training)
            x = self.leaky_relu_layers[i](x)

        skip_connection_conv = x # La salida del bloque convolucional es la conexión skip

        # Preparar entrada para LSTM
        x_lstm_processed = skip_connection_conv # Por defecto, si no hay LSTMs
        if self.lstm_layers:
            current_x_for_lstm = skip_connection_conv
            batch_size = tf.shape(current_x_for_lstm)[0]
            
            if self.data_format == 'channels_last':
                # Forma de x: (batch, H_c, W_c, C_c)
                h_conv_out = tf.shape(current_x_for_lstm)[1]
                w_conv_out = tf.shape(current_x_for_lstm)[2]
                self.original_spatial_dims_for_lstm = (h_conv_out, w_conv_out)
                num_channels_conv = tf.shape(current_x_for_lstm)[3]
                # Reformar a (batch, H_c * W_c, C_c)
                x_lstm_input_reshaped = tf.reshape(current_x_for_lstm, [batch_size, h_conv_out * w_conv_out, num_channels_conv])
            else: # channels_first
                # Forma de x: (batch, C_c, H_c, W_c)
                num_channels_conv = tf.shape(current_x_for_lstm)[1]
                h_conv_out = tf.shape(current_x_for_lstm)[2]
                w_conv_out = tf.shape(current_x_for_lstm)[3]
                self.original_spatial_dims_for_lstm = (h_conv_out, w_conv_out)
                x_permuted = tf.transpose(current_x_for_lstm, [0, 2, 3, 1]) # (batch, H_c, W_c, C_c)
                x_lstm_input_reshaped = tf.reshape(x_permuted, [batch_size, h_conv_out * w_conv_out, num_channels_conv])

            # Aplicar capas LSTM
            current_call_states = []
            lstm_temp_output = x_lstm_input_reshaped
            for i, lstm_layer in enumerate(self.lstm_layers):
                initial_state_for_layer = None
                if self.pending_initial_lstm_states and self.pending_initial_lstm_states[i] is not None:
                    initial_state_for_layer = self.pending_initial_lstm_states[i]
                
                lstm_temp_output = lstm_layer(lstm_temp_output, training=training, initial_state=initial_state_for_layer)
                current_call_states.append(lstm_layer.states) 
            
            self.current_lstm_states = current_call_states
            if self.pending_initial_lstm_states:
                self.pending_initial_lstm_states = None

            # Reformar la salida LSTM de vuelta a (batch_size, height, width, channels)
            lstm_output_units = self.lstm_layers[-1].units
            if self.data_format == 'channels_last':
                x_lstm_processed = tf.reshape(lstm_temp_output, [batch_size, self.original_spatial_dims_for_lstm[0], self.original_spatial_dims_for_lstm[1], lstm_output_units])
            else: # channels_first
                x_reshaped_channels_last = tf.reshape(lstm_temp_output, [batch_size, self.original_spatial_dims_for_lstm[0], self.original_spatial_dims_for_lstm[1], lstm_output_units])
                x_lstm_processed = tf.transpose(x_reshaped_channels_last, [0, 3, 1, 2])
        
        return x_lstm_processed, skip_connection_conv

    def reset_states_per_batch(self, is_last_batch):
        if is_last_batch:
            for lstm_layer in self.lstm_layers:
                lstm_layer.reset_states()
            self.current_lstm_states = [None] * len(self.lstm_layers)

    def get_states(self):
        return self.current_lstm_states

    def set_states(self, states):
        if len(states) == len(self.lstm_layers):
            self.pending_initial_lstm_states = states
        else:
            raise ValueError("El número de estados proporcionados no coincide con el número de capas LSTM.")


class UpBlock(tf.keras.Model):
    def __init__(self, conv_kernels, up_factor, data_format='channels_last', name='up_block'):
        """
        Inicializa el UpBlock.
        Args:
            conv_kernels: Lista de diccionarios para capas Conv2D.
            up_factor: Tupla (factor_altura, factor_ancho) para upsampling.
            data_format: 'channels_first' o 'channels_last'.
        """
        super(UpBlock, self).__init__(name=name)
        self.up_sampling_layer = tf.keras.layers.UpSampling2D(
            size=up_factor, interpolation='bilinear', data_format=data_format, name=f"{name}_upsample"
        )
        self.concat_layer = tf.keras.layers.Concatenate(
            axis=-1 if data_format == 'channels_last' else 1, name=f"{name}_concat"
        )
        
        self.conv_layers = []
        self.bn_layers = [] # Agregando Batch Norm también en UpBlock por consistencia
        self.leaky_relu_layers = []
        self.data_format = data_format
        self.bn_axis = -1 if data_format == 'channels_last' else 1

        for i, conv_params in enumerate(conv_kernels):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=conv_params['filters'],
                    kernel_size=conv_params['kernel_size'],
                    strides=conv_params.get('strides', 1),
                    padding=conv_params.get('padding', 'same'),
                    data_format=data_format,
                    name=f"{name}_conv2d_{i}"
                )
            )
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(axis=self.bn_axis, name=f"{name}_bn_{i}")
            )
            self.leaky_relu_layers.append(
                tf.keras.layers.LeakyReLU(alpha=0.2, name=f"{name}_leakyrelu_{i}")
            )

    def call(self, inputs, skip_connection, training=False):
        """
        Pase hacia adelante para el UpBlock.
        Args:
            inputs: Tensor de entrada (de la capa inferior previa).
            skip_connection: Tensor de conexión skip (del DownBlock correspondiente).
            training: Booleano.
        Returns:
            Tensor procesado.
        """
        x = self.up_sampling_layer(inputs)
        
        # Asegurar que las dimensiones espaciales coincidan para la concatenación
        # Esto es crucial. Una forma robusta es redimensionar skip_connection a las dims de x si es necesario.
        # Ejemplo:
        if self.data_format == 'channels_last':
            target_h, target_w = tf.shape(x)[1], tf.shape(x)[2]
            if tf.shape(skip_connection)[1] != target_h or tf.shape(skip_connection)[2] != target_w:
                skip_connection = tf.image.resize(skip_connection, [target_h, target_w])
        else: # channels_first
            target_h, target_w = tf.shape(x)[2], tf.shape(x)[3]
            if tf.shape(skip_connection)[2] != target_h or tf.shape(skip_connection)[3] != target_w:
                # Transponer, redimensionar, y transponer de vuelta
                skip_connection_transposed = tf.transpose(skip_connection, [0, 2, 3, 1]) # B, H, W, C
                skip_connection_resized = tf.image.resize(skip_connection_transposed, [target_h, target_w])
                skip_connection = tf.transpose(skip_connection_resized, [0, 3, 1, 2]) # B, C, H, W


        x = self.concat_layer([x, skip_connection])

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x, training=training)
            x = self.leaky_relu_layers[i](x)
        return x


class ULSTMnet2D(tf.keras.Model):
    def __init__(self, output_channels=4, data_format='channels_last', name='ulstm_unet_2d_4out_explicit'):
        """
        Inicializa ULSTMnet2D con una estructura de red explícita.
        Filtros: 32, 64, 128, 256, 512 para downsampling.
        Args:
            output_channels: Número de canales de la imagen de salida (4 en este caso).
            data_format: 'channels_first' o 'channels_last'.
        """
        super(ULSTMnet2D, self).__init__(name=name)
        self.data_format = data_format
        common_conv_kernel_size = 3
        common_lstm_num_layers = 1 # Una LSTM por DownBlock como en el ejemplo anterior

        # --- Bloques de Downsampling (Encoder) ---
        # D0 (Entrada: 256x64 -> Salida Conv: 128x32x32, Salida LSTM: 128x32xUnitsLSTM)
        self.down_block_0 = DownBlock(
            conv_kernels=[{'filters': 32, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 32, 'kernel_size': common_conv_kernel_size}],
            lstm_params_list=[{'units': 32}] * common_lstm_num_layers,
            stride=2, data_format=data_format, name=f"{name}_down_block_0"
        )
        # D1 (128x32 -> 64x16x64)
        self.down_block_1 = DownBlock(
            conv_kernels=[{'filters': 64, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 64, 'kernel_size': common_conv_kernel_size}],
            lstm_params_list=[{'units': 64}] * common_lstm_num_layers,
            stride=2, data_format=data_format, name=f"{name}_down_block_1"
        )
        # D2 (64x16 -> 32x8x128)
        self.down_block_2 = DownBlock(
            conv_kernels=[{'filters': 128, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 128, 'kernel_size': common_conv_kernel_size}],
            lstm_params_list=[{'units': 128}] * common_lstm_num_layers,
            stride=2, data_format=data_format, name=f"{name}_down_block_2"
        )
        # D3 (32x8 -> 16x4x256)
        self.down_block_3 = DownBlock(
            conv_kernels=[{'filters': 256, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 256, 'kernel_size': common_conv_kernel_size}],
            lstm_params_list=[{'units': 256}] * common_lstm_num_layers,
            stride=2, data_format=data_format, name=f"{name}_down_block_3"
        )
        # D4 (Cuello de botella: 16x4 -> 8x2x512)
        self.down_block_4_bottleneck = DownBlock(
            conv_kernels=[{'filters': 512, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 512, 'kernel_size': common_conv_kernel_size}],
            lstm_params_list=[{'units': 512}] * common_lstm_num_layers,
            stride=2, data_format=data_format, name=f"{name}_down_block_4_bottleneck"
        )

        # --- Bloques de Upsampling (Decoder) ---
        # U0 (Entrada LSTM de D4: 8x2x512, Skip de Conv D3: 16x4x256 -> Salida: 16x4x256)
        self.up_block_0 = UpBlock(
            conv_kernels=[{'filters': 256, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 256, 'kernel_size': common_conv_kernel_size}],
            up_factor=(2, 2), data_format=data_format, name=f"{name}_up_block_0"
        )
        # U1 (16x4x256, Skip D2: 32x8x128 -> 32x8x128)
        self.up_block_1 = UpBlock(
            conv_kernels=[{'filters': 128, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 128, 'kernel_size': common_conv_kernel_size}],
            up_factor=(2, 2), data_format=data_format, name=f"{name}_up_block_1"
        )
        # U2 (32x8x128, Skip D1: 64x16x64 -> 64x16x64)
        self.up_block_2 = UpBlock(
            conv_kernels=[{'filters': 64, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 64, 'kernel_size': common_conv_kernel_size}],
            up_factor=(2, 2), data_format=data_format, name=f"{name}_up_block_2"
        )
        # U3 (64x16x64, Skip D0: 128x32x32 -> 128x32x32)
        self.up_block_3 = UpBlock(
            conv_kernels=[{'filters': 32, 'kernel_size': common_conv_kernel_size}, 
                          {'filters': 32, 'kernel_size': common_conv_kernel_size}],
            up_factor=(2, 2), data_format=data_format, name=f"{name}_up_block_3"
        )
        
        # Capa de salida final para generar las 4 imágenes
        self.final_up_sampling_layer = tf.keras.layers.UpSampling2D(
            size=(2,2), interpolation='bilinear', 
            data_format=data_format, name=f"{name}_final_upsample"
        ) # Para ir de 128x32 a 256x64 (si la salida de U3 es 128x32x32)
        self.output_conv_layer = tf.keras.layers.Conv2D(
            filters=output_channels, 
            kernel_size=3, # O 1 para una transformación final simple
            padding='same',
            activation='sigmoid', # Asumiendo salida normalizada a [0,1]
            data_format=data_format,
            name=f"{name}_output_conv_{output_channels}images"
        )

    def call(self, inputs, training=False):
        """
        Pase hacia adelante para ULSTMnet2D.
        Args:
            inputs: Tensor de entrada.
            training: Booleano.
        Returns:
            Tensor de salida (las 4 imágenes).
        """
        # --- Recorrido hacia abajo (Encoder) ---
        s0_lstm, s0_conv_skip = self.down_block_0(inputs, training=training)
        s1_lstm, s1_conv_skip = self.down_block_1(s0_lstm, training=training)
        s2_lstm, s2_conv_skip = self.down_block_2(s1_lstm, training=training)
        s3_lstm, s3_conv_skip = self.down_block_3(s2_lstm, training=training)
        
        # Cuello de botella
        bottleneck_lstm, _ = self.down_block_4_bottleneck(s3_lstm, training=training) 
        # La salida convolucional del cuello de botella no se usa típicamente como skip directo 
        # para el primer UpBlock si el UpBlock toma la salida LSTM del cuello de botella.

        # --- Recorrido hacia arriba (Decoder) ---
        u0 = self.up_block_0(bottleneck_lstm, s3_conv_skip, training=training) # Skip de D3
        u1 = self.up_block_1(u0, s2_conv_skip, training=training)                # Skip de D2
        u2 = self.up_block_2(u1, s1_conv_skip, training=training)                # Skip de D1
        u3 = self.up_block_3(u2, s0_conv_skip, training=training)                # Skip de D0
        
        # Etapa de salida final
        final_features = self.final_up_sampling_layer(u3)
        output_images = self.output_conv_layer(final_features)
        
        return output_images

    def reset_states_per_batch(self, is_last_batch):
        self.down_block_0.reset_states_per_batch(is_last_batch)
        self.down_block_1.reset_states_per_batch(is_last_batch)
        self.down_block_2.reset_states_per_batch(is_last_batch)
        self.down_block_3.reset_states_per_batch(is_last_batch)
        self.down_block_4_bottleneck.reset_states_per_batch(is_last_batch)

    def get_states(self):
        return [
            self.down_block_0.get_states(),
            self.down_block_1.get_states(),
            self.down_block_2.get_states(),
            self.down_block_3.get_states(),
            self.down_block_4_bottleneck.get_states()
        ]

    def set_states(self, states_for_all_down_blocks):
        if len(states_for_all_down_blocks) == 5: # 5 DownBlocks
            self.down_block_0.set_states(states_for_all_down_blocks[0])
            self.down_block_1.set_states(states_for_all_down_blocks[1])
            self.down_block_2.set_states(states_for_all_down_blocks[2])
            self.down_block_3.set_states(states_for_all_down_blocks[3])
            self.down_block_4_bottleneck.set_states(states_for_all_down_blocks[4])
        else:
            raise ValueError("El número de listas de estados (se esperan 5) no coincide con el número de DownBlocks.")

    def build_graph(self, input_shape_no_batch):
        if len(input_shape_no_batch) != 3:
            raise ValueError("input_shape_no_batch debe ser (H, W, C) o (C, H, W)")
        
        if self.data_format == 'channels_last':
            input_shape_with_batch = (None,) + input_shape_no_batch 
        else: 
            input_shape_with_batch = (None, input_shape_no_batch[0], input_shape_no_batch[1], input_shape_no_batch[2])
        
        dummy_input = tf.keras.Input(shape=input_shape_with_batch[1:], batch_size=input_shape_with_batch[0])
        _ = self.call(dummy_input) 
        print(f"Grafo del modelo '{self.name}' construido.")


if __name__ == '__main__':
    data_format_choice = 'channels_last' 
    input_h, input_w, input_c = 256, 64, 1
    num_output_images = 4

    if data_format_choice == 'channels_last':
        input_shape_for_model = (input_h, input_w, input_c) 
    else: # channels_first
        input_shape_for_model = (input_c, input_h, input_w)

    # Crear el modelo
    unet_model_explicit = ULSTMnet2D(output_channels=num_output_images, data_format=data_format_choice)
    
    # Construir el grafo y ver resumen
    unet_model_explicit.build_graph(input_shape_for_model) 
    unet_model_explicit.summary(line_length=150) # Aumentar line_length para mejor visualización del resumen


    # --- Probar con datos dummy ---
    batch_s = 2
    if data_format_choice == 'channels_last':
        dummy_tensor = tf.random.normal((batch_s, input_h, input_w, input_c))
    else:
        dummy_tensor = tf.random.normal((batch_s, input_c, input_h, input_w))

    print(f"\nTensor de entrada dummy: {dummy_tensor.shape}")

    # --- Probar el pase hacia adelante ---
    print("Procesando una secuencia de ejemplo...")
    output_images_s1 = unet_model_explicit(dummy_tensor, training=False)
    print(f"Forma de las imágenes de salida (secuencia 1): {output_images_s1.shape}")
    
    # Resetear estados LSTM (simulando fin de lote/secuencia)
    unet_model_explicit.reset_states_per_batch(is_last_batch=True)
    print("Estados LSTM reseteados.")

    output_images_s2 = unet_model_explicit(dummy_tensor, training=False) # Misma entrada para simplicidad
    print(f"Forma de las imágenes de salida (secuencia 2, estados frescos): {output_images_s2.shape}")

    # --- Probar get_states y set_states (ejemplo conceptual) ---
    print("\nProbando get_states y set_states...")
    # Procesar una vez para tener estados
    _ = unet_model_explicit(dummy_tensor, training=False)
    current_states = unet_model_explicit.get_states()
    # print(f"Estados obtenidos: {current_states}") # Serán tensores simbólicos o valores si se evalúan
    
    # Crear estados dummy para setear (deben tener la estructura correcta [h,c] por cada LSTM)
    # Esto es solo conceptual, ya que crear estados válidos manualmente es complejo.
    # En un caso real, estos serían estados guardados de una ejecución previa.
    # Por ahora, reseteamos y luego verificamos que el procesamiento es "fresco"
    unet_model_explicit.reset_states_per_batch(is_last_batch=True) 
    # Si tuviéramos 'previous_valid_states', podríamos hacer:
    # unet_model_explicit.set_states(previous_valid_states)
    # Y la siguiente llamada usaría esos estados iniciales.

    print("\nEjemplo de ejecución finalizado.")
