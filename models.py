import keras
import datetime
import tensorflow                       as tf
from tensorflow.keras.callbacks         import TensorBoard
from tensorflow.keras.layers            import Input,Lambda,UpSampling2D,Conv2D,Dropout,MaxPooling2D,Conv2DTranspose,concatenate,Flatten,BatchNormalization, Activation,ConvLSTM2D,TimeDistributed, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, multiply,Reshape,RepeatVector,LayerNormalization
from tensorflow.keras.models            import Model
from tensorflow.keras.optimizers        import Adam,RMSprop,SGD
from keras.utils                        import plot_model
from tensorflow.keras                   import layers, models
from tensorflow.keras.losses            import mae
from tensorflow.keras.callbacks         import LearningRateScheduler,Callback,CSVLogger

from sklearn.metrics                    import mean_squared_error,r2_score,mean_absolute_error

import sys
import os
import numpy as np
import math
import time
import random, time
from pathlib                        import Path
from PIL                            import Image

import skimage                      as ski
from   skimage.filters              import threshold_otsu
from   skimage                      import io, color
from   skimage.color                import rgb2gray
from   skimage                      import filters
import cv2                          as cv
import matplotlib.pyplot            as plt 
import gc
import glob
from skimage                        import img_as_ubyte
from skimage                        import io
import shutil
import pandas as pd
tf.keras.backend.clear_session()

type_padding = 'same'
f_activation = 'relu'
f_activation_last='relu'

# *******  Unet  *******
number_of_filters_15 = [32,64,128,256,512]

def conv_block_batchnorm(filters,x):
    conv = Conv2D(filters, (3, 3), padding=type_padding)(x)
    conv= BatchNormalization()(conv)
    conv = Activation(f_activation)(conv)
    conv = Conv2D(filters, (3, 3), padding=type_padding)(conv)
    conv= BatchNormalization()(conv)
    conv = Activation(f_activation)(conv)
    return conv
    
def conv_block(filters,x):
    conv = Conv2D(filters, (3, 3), activation=f_activation, padding=type_padding)(x)
    conv = Conv2D(filters, (3, 3), activation=f_activation, padding=type_padding)(conv)
    return conv    

def encoder(x,filters):
    conv = conv_block_batchnorm(filters,x)
    downsample = MaxPooling2D((2,2))(conv)
    return conv,downsample


def decoder(x1,x2,filters,transpose=None):
    if transpose != None:
        conv_up = Conv2DTranspose(filters,(2,2),strides=(2, 2),padding=type_padding)(x1)
    else:
        conv_up = UpSampling2D((2, 2))(x1)
   
    concat=concatenate([conv_up,x2],axis = 3)
    up = conv_block(filters, concat)
    return up

# ******* U-Net-CAM-SAM *******
number_of_filters_14 = [32,64,128,256,512]
def channel_attention_module(input_tensor,ratio=8):
    
    channel_axis = -1
    channels = input_tensor.shape[channel_axis]
    
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = GlobalMaxPooling2D()(input_tensor)

    mlp = tf.keras.Sequential([
        layers.Dense(channels // ratio, activation='relu'),
        layers.Dense(channels)
    ])
    
    avg_out = mlp(avg_pool)
    max_out = mlp(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    cbam_feature = Reshape((1, 1, channels))(cbam_feature)

    return multiply([input_tensor, cbam_feature])

def spatial_attention_module(input_tensor):
      kernel_size = 7
      avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(input_tensor)
      max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(input_tensor)
      concat = concatenate([avg_pool, max_pool], axis=-1)
      conv = Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(concat)
      return  multiply([input_tensor, conv])

def decoder_attention(x,skip,filters,transpose=None): ### x2 is the skip
    if transpose != None:
        conv_up = Conv2DTranspose(filters,(2,2),strides=(2, 2),padding=type_padding)(x)
    else:
        conv_up = UpSampling2D((2, 2))(x)
    sam = spatial_attention_module(skip)
    concat=concatenate([conv_up,sam],axis = 3)
    up = conv_block(filters, concat)
    return up

# ****** U-Net-ConvLSTM *******
number_of_filters_13 = [64,128,256,512]

def conv2d1(x,filters,number):
	x=Conv2D(filters, (1, 1), activation='relu', padding=type_padding, name=f'K_{number}')(x)
	return x

def encoder_convlstm(x,filters,number):
    conv = conv_block_batchnorm(filters,x)
    conv = conv2d1(conv,filters,number)
    downsample = MaxPooling2D((2,2))(conv)
    return conv,downsample

def convLSTM_block(filters,x):	
    x_reshape= Lambda(lambda xx: tf.reshape(xx, (tf.shape(xx)[0],1,tf.shape(xx)[1], tf.shape(xx)[2], tf.shape(xx)[3])))(x)
    conv_lstm = ConvLSTM2D(filters, (3, 3), padding=type_padding, return_sequences=True)(x_reshape)
    conv_lstm = ConvLSTM2D(filters, (3, 3), padding=type_padding, return_sequences=True)(conv_lstm)
    conv_lstm = ConvLSTM2D(filters, (3, 3), padding=type_padding, return_sequences=False)(conv_lstm)
    return conv_lstm 

# ******* U-Net-BILSTM *******
number_of_filters_12 = [32,64,128,256,512]
units = [16,32,64,128]
def biLSTM(x,units,prefix_name):
    B,h,w,c=x.shape
    # Reshape (B, H, W, C) -> (B, W, H*C) for column-wise LSTM
  
    x_seq = Lambda(lambda xx: tf.reshape(xx, (tf.shape(xx)[0], tf.shape(xx)[2], tf.shape(xx)[1] * tf.shape(xx)[3])))(x)
    x_bilstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True,name=f"{prefix_name}_bilstm"))(x_seq) 
    x_dense = layers.TimeDistributed(layers.Dense(h * c, activation='relu', name=f'{prefix_name}_dense'),name=f'{prefix_name}_timedist_dense')(x_bilstm)
    #x_dense = layers.Dense(h*c)(x_bilstm)
    output_tensor = layers.Reshape((h,w,c))( x_dense)

    return output_tensor

#*******************************************  MODELS *******************************************************************
#     1) U-Net  (new15)

def unet(img_height, img_width, channel,model_name,optimizer):
    
    image_input = Input((img_height, img_width, channel))
    
    conv1,down_block1 = encoder (image_input, number_of_filters_15[0])
    conv2,down_block2 = encoder (down_block1 , number_of_filters_15[1])
    conv3,down_block3 = encoder (down_block2 , number_of_filters_15[2])
    conv4,down_block4 = encoder (down_block3 , number_of_filters_15[3])

    conv5 = conv_block(number_of_filters_15[4],down_block4)
   
 
    up6=decoder(conv5,conv4,number_of_filters_15[3],transpose='yes')
    up7=decoder(up6,conv3,number_of_filters_15[2],transpose='yes')
    up8=decoder(up7,conv2,number_of_filters_15[1],transpose='yes')
    up9=decoder(up8,conv1,number_of_filters_15[0],transpose='yes')

    flows_out = Conv2D(4, (1, 1), activation=f_activation_last,name='flows_output',padding="same")(up9)

    # construct model
    model =  keras.Model(inputs=image_input, outputs=[flows_out],name= model_name)
    
    model.summary()

    model.compile(optimizer=optimizer, 
              loss = 'mean_squared_error',
              
              metrics= ['mae', "root_mean_squared_error"] )

    return model  

# 2) unet-cam-sam (new14)

def unet_cam_sam(img_height, img_width, channel,model_name,optimizer):

    image_input = Input((img_height, img_width, channel))
    
    conv1,down_block1 = encoder (image_input, number_of_filters_14[0])
    conv2,down_block2 = encoder (down_block1 , number_of_filters_14[1])
    conv3,down_block3 = encoder (down_block2 , number_of_filters_14[2])
    conv4,down_block4 = encoder (down_block3 , number_of_filters_14[3])

    conv5 = conv_block(number_of_filters_14[4],down_block4)
    bottleneck = channel_attention_module(conv5)
   
 
    up6=decoder(bottleneck,conv4,number_of_filters_14[3],transpose='yes')
    up7=decoder(up6,conv3,number_of_filters_14[2],transpose='yes')
    up8=decoder(up7,conv2,number_of_filters_14[1],transpose='yes')
    up9=decoder(up8,conv1,number_of_filters_14[0],transpose='yes')

    flows_out = Conv2D(4, (1, 1), activation=f_activation_last,name='flows_output',padding="same")(up9)

    # construct model
    model =  keras.Model(inputs=image_input, outputs=[flows_out],name= model_name)
    
    model.summary()

    model.compile(optimizer=optimizer, 
              loss = 'mean_squared_error',
              
              metrics= ['mae', "root_mean_squared_error"] )

    return model  

# 3) unet-convLSTM (new13)

def unet_convlstm(img_height, img_width, channel,model_name,optimizer):
    image_input = Input((img_height, img_width, channel)) 
    s1,x1 =   encoder(image_input,number_of_filters_13[0],1) 
    s2,x2 =   encoder(x1,number_of_filters_13[1],2) 
    s3,x3 =   encoder(x2,number_of_filters_13[2],3) 
    s4,x4 =   encoder(x3,number_of_filters_13[3],4) 
	
    x5 = conv_block(number_of_filters_13[3],x4)
    x6 = convLSTM_block(number_of_filters_13[3],x5)
	
    up7=decoder(x5,s4,number_of_filters_13[3],transpose='yes')
    up8=decoder(up7,s3,number_of_filters_13[2],transpose='yes')
    up9=decoder(up8,s2,number_of_filters_13[1],transpose='yes')
    up10=decoder(up9,s1,number_of_filters_13[0],transpose='yes')
	
    flows_out = Conv2D(4, (1, 1), activation=f_activation_last,name='flows_output',padding="same")(up10)
	
		 # construct model
    model =  keras.Model(inputs=image_input, outputs=[flows_out],name= model_name)
	    
    model.summary()
   
    model.compile(optimizer=optimizer, 
              loss = 'mean_squared_error',
              
              metrics= ['mae', "root_mean_squared_error"] )

    return model  

# 4) unet-BiLSTM (new12)

def unet_BiLSTM(img_height, img_width, channel,model_name,optimizer):
    
    image_input = Input((img_height, img_width, channel))
    
    conv1,down_block1 = encoder (image_input, number_of_filters_12[0])
    biconv1=biLSTM(conv1,units[0],1)
    
    conv2,down_block2 = encoder (down_block1 , number_of_filters_12[1])
    biconv2=biLSTM(conv2,units[1],2)
    
    conv3,down_block3 = encoder (down_block2 , number_of_filters_12[2])
    biconv3=biLSTM(conv3,units[2],3)
    
    conv4,down_block4 = encoder (down_block3 , number_of_filters_12[3])
    biconv4=biLSTM(conv4,units[3],4)

    conv5 = conv_block(number_of_filters_12[4],down_block4)
   
    print('conv5',conv5)
    print('biconv4',biconv4)
    
    up6=decoder(conv5,biconv4,number_of_filters_12[3],transpose='yes')
    up7=decoder(up6,biconv3,number_of_filters_12[2],transpose='yes')
    up8=decoder(up7,biconv2,number_of_filters_12[1],transpose='yes')
    up9=decoder(up8,biconv1,number_of_filters_12[0],transpose='yes')

    flows_out = Conv2D(4, (1, 1), activation=f_activation_last,name='flows_output',padding="same")(up9)

    # construct model
    model =  keras.Model(inputs=image_input, outputs=[flows_out],name= model_name)
    
    model.summary()

    model.compile(optimizer=optimizer, 
              loss = 'mean_squared_error',
              
              metrics= ['mae', "root_mean_squared_error"] )

    return model  

# 5) convlstm (new16)

def convlstm(img_height, img_width, channel,model_name,optimizer):
    
    image_input = Input((img_height, img_width, channel))
    reshaped_input = layers.Reshape((1,img_height, img_width, channel))(image_input)

    x1 = layers.ConvLSTM2D(filters=8, kernel_size=3, padding="same", return_sequences=True)(reshaped_input)
    x1 = layers.BatchNormalization()(x1)
    x2 = layers.ConvLSTM2D(filters=16, kernel_size=3, padding="same", return_sequences=True)(x1)
    x2 = layers.BatchNormalization()(x2)
    x3 = layers.ConvLSTM2D(filters=32, kernel_size=3, padding="same", return_sequences=True)(x2)
    x3= layers.BatchNormalization()(x3)
    
    bottleneck = layers.ConvLSTM2D(filters=64, kernel_size=3, padding="same", return_sequences=True)(x3)
    bottleneck = layers.BatchNormalization()(bottleneck)

    d1 = layers.ConvLSTM2D(filters=32, kernel_size=3, padding="same", return_sequences=True)(bottleneck)
    d1 = layers.Concatenate()([d1, x3])
    d1 = layers.BatchNormalization()(d1)

    d2 = layers.ConvLSTM2D(filters=16, kernel_size=3, padding="same", return_sequences=True)(d1)
    d2 = layers.Concatenate()([d2,x2])
    d2 = layers.BatchNormalization()(d2)

    d3 = layers.ConvLSTM2D(filters=8, kernel_size=3, padding="same", return_sequences=True)(d2)
    d3 = layers.Concatenate()([d3,x1])
    d3 = layers.BatchNormalization()(d3)

  
    temporal = layers.Conv3D(4, (1, 1,1), activation=f_activation_last,name='flows_output',padding="same")(d3)
    # construct model
    # Squeeze for removing temporal axis: (None, 1, 64, 256, 4) -> (None, 64, 256, 4)
    flows_out= layers.Lambda(lambda x: tf.squeeze(x, axis=1))(temporal)
  
    model =  keras.Model(inputs=image_input, outputs=[flows_out],name= model_name)
    
    model.summary()

  
    model.compile(optimizer=optimizer, 
              loss = 'mean_squared_error',
              
              metrics= ['mae', "root_mean_squared_error"] )

    return model  
