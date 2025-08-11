import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot            as plt 
import os
import sys
import yaml

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
import gc
import glob
from skimage                        import img_as_ubyte
from skimage                        import io
import shutil
tf.keras.backend.clear_session()

from models import *
from NewData import *

with open("config.yaml") as f:
    config = yaml.safe_load(f)

directory= os.path.abspath(config['data']['path_directory'])
sys.path.append(directory)


inp=config['data']['inp_path']
out=config['data']['out_path']
ntrain=config['data']['ntrain']
ntest=config['data']['ntest']
xtrain,xval,xtest,ytrain,yval,ytest=split_data(inp,out,ntrain,ntest)

# Model name
model_name = 'Unet_CLASSIC'
NamesKeras  = 'Arch_1.keras'
SaveModel = 'unet.keras'
# image dimensions
img_width     =  config['model']['img_width']
img_height    =  config['model']['img_height']
channel       =  config['model']['channel']

def exponential_decay(epoch,lr_ini=config['training']['LR'],decay_rate=config['training']['DECAY_RATE'],epochs=config['training']['epochs']):
    if epoch < epochs*0.001:
        return lr_ini
    else:
        return  lr_ini * np.exp(-decay_rate*epoch) 
    
optimizer = Adam(learning_rate=config['training']['LR'])

checkpoint_path='/Unet_CLASSIC_weights/Test5_Best_weights.weights.h5'

checkpoint_file=config['save']['save_in']+checkpoint_path

callbacks = [
    LearningRateScheduler(lambda epoch: exponential_decay(epoch), verbose=1),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_file,verbose=1,save_weights_only=True),
    tf.keras.callbacks.ModelCheckpoint(NamesKeras,verbose=1,save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                    patience=config['training']['patience'],
                                    restore_best_weights=True,
                                    verbose=1),
    tf.keras.callbacks.CSVLogger(
        model_name+"/training.csv", 
        separator = ',', 
        append = False
    )
    ]

model = unet(img_height, img_width, channel,model_name,optimizer)

print("Starting training")
start_time_for_fit = time.time()

history = model.fit(
                    xtrain,
                    epochs=config['training']['epochs'],
                    verbose=1,
                    validation_data = xval,
                    callbacks=callbacks)

end_time_for_fit = time.time()
Training_time = end_time_for_fit - start_time_for_fit
print(f"TRAINING TIME IN  {Training_time:.4f} segundos.")

model.save(SaveModel)
unet_history_df = pd.DataFrame(history.history)

csv_name = "unet.csv"

with open(csv_name, mode='w') as f:
    unet_history_df.to_csv(f)



