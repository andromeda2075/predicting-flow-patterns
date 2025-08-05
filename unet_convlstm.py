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



LR=0.001
DECAY_RATE=0.04
num_epochs    =200
model_names=['Unet_ConvLSTM']
model_name=model_names[0]

def exponential_decay(epoch,lr_ini=LR,decay_rate=DECAY_RATE,epochs=num_epochs):
    if epoch < epochs*0.001:
        return lr_ini
    else:
        return  lr_ini * np.exp(-decay_rate*epoch)
    

img_width     =  256   # 739   G:737
img_height    =  64   # 185
channel       =  1
number_of_filters = [64,128,256,512]

type_padding = 'same'
f_activation = 'relu'
f_activation_last='relu'

optimizer = Adam(learning_rate=LR)


def conv_block_batchnorm(filters,x):
    conv = Conv2D(filters, (3, 3), padding=type_padding)(x)
    conv= BatchNormalization()(conv)
    conv = Activation(f_activation)(conv)
    conv = Conv2D(filters, (3, 3), padding=type_padding)(conv)
    conv= BatchNormalization()(conv)
    conv = Activation(f_activation)(conv)
    return conv
    
def conv2d1(x,filters,number):
	x=Conv2D(filters, (1, 1), activation='relu', padding=type_padding, name=f'K_{number}')(x)
	return x
	    
def conv_block(filters,x):
    conv = Conv2D(filters, (3, 3), activation=f_activation, padding=type_padding)(x)
    conv = Conv2D(filters, (3, 3), activation=f_activation, padding=type_padding)(conv)
    return conv    
    
def encoder(x,filters,number):
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

def decoder(x1,x2,filters,transpose=None):
    if transpose != None:
        conv_up = Conv2DTranspose(filters,(2,2),strides=(2, 2),padding=type_padding)(x1)
    else:
        conv_up = UpSampling2D((2, 2))(x1)
   
    concat=concatenate([conv_up,x2],axis = 3)
    up = conv_block(filters, concat)
    return up

def make_model():
    image_input = Input((img_height, img_width, channel)) 
    s1,x1 =   encoder(image_input,number_of_filters[0],1) 
    s2,x2 =   encoder(x1,number_of_filters[1],2) 
    s3,x3 =   encoder(x2,number_of_filters[2],3) 
    s4,x4 =   encoder(x3,number_of_filters[3],4) 
	
    x5 = conv_block(number_of_filters[3],x4)
    x6 = convLSTM_block(number_of_filters[3],x5)
	
    up7=decoder(x5,s4,number_of_filters[3],transpose='yes')
    up8=decoder(up7,s3,number_of_filters[2],transpose='yes')
    up9=decoder(up8,s2,number_of_filters[1],transpose='yes')
    up10=decoder(up9,s1,number_of_filters[0],transpose='yes')
	
    flows_out = Conv2D(4, (1, 1), activation=f_activation_last,name='flows_output',padding="same")(up10)
	
		 # construct model
    model =  keras.Model(inputs=image_input, outputs=[flows_out],name= model_name)
	    
    model.summary()
   
    model.compile(optimizer=optimizer, 
              loss = 'mean_squared_error',
              
              metrics= ['mae', "root_mean_squared_error"] )

    return model  
