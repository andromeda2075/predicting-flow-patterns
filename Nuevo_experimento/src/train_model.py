import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import sys
import yaml
from pathlib import Path
import keras
import datetime
from datetime import timedelta
import time
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
from utils import *
from config import load_config
from dataset import load_dataset
from u_net_architecture import *
import json
from tensorflow.keras import backend as K

#================== Uso progresivo de VRAM ====================================

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ============ Load Config and Dataset ================
config = load_config()
train,val,test=load_dataset()

#========== Save results of the model in =================
def format_save_in(name_output_file):
    #/home/sade06cy/predicting-flow-patterns/New_Experiment/Results_training/U_Net_Mlp
    return os.path.join(config['dir_save'],config['model_name'],name_output_file) 


# ============ Format time ====================
def save_format(model,train_time):
    ruta_txt=format_save_in('train_time.txt')
    duracion = str(timedelta(seconds=round(train_time)))
    with open(ruta_txt, "w") as archivo:
        archivo.write(f"Parameters: {model.count_params()}, Tiempo segundos: {train_time:.4f}, Formato de tiempo {duracion} \n")

# ==========  Callbacks ===============================
def exponential_decay(epoch,lr_ini=config['lr_init'],decay_rate=config['dr'],epochs=config['epoch']):
    if epoch < epochs*0.001:
        return lr_ini
    else:
        return  lr_ini * np.exp(-decay_rate*epoch)
    
def callbacks(dir_save_weight):
    lr_scheduler = LearningRateScheduler(lambda epoch: exponential_decay(config['epoch']), verbose=1)
    optimizer = Adam(learning_rate=config['lr_init'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'], restore_best_weights=True,verbose=1)
    #os.makedirs(dir_save_weight)
    #folder_path = Path(config['model_name'])
    #folder_path.mkdir(parents=True, exist_ok=True)
    #file_path = folder_path / "training_log.csv"
    
    create_folder_results=Path(os.path.join(config['dir_save'],config['model_name']))
    create_folder_results.mkdir(parents=True, exist_ok=True)

    name_best_weight=config['model_name']+'_'+ 'best.weights.h5'

    checkpoint_weight=tf.keras.callbacks.ModelCheckpoint(format_save_in(name_best_weight),save_weights_only=True)
    csv_logger = CSVLogger(format_save_in("training_log.csv"), append=False)

    return lr_scheduler,optimizer, early_stopping,checkpoint_weight,csv_logger

def compile_moldel(model,optimizer):

    model.compile(

              #optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr_init']), 
              optimizer=optimizer,
              loss = {
                        'flow_output': 'mse',
                        'drag_1': 'mse',
                        'lift_1': 'mse',
                        'drag_2': 'mse',
                        'lift_2': 'mse'
                    },
              loss_weights = {
                        "flow_output":1.0 - config['lambda'],
                        "drag_1":0.25* config['lambda'],
                        "lift_1":0.25* config['lambda'],
                        "drag_2":0.25* config['lambda'],
                        "lift_2":0.25* config['lambda']},
              
             metrics= {
                        'flow_output': ['mae'], # tf.keras.metrics.RootMeanSquaredError()
                        'drag_1':['mae'],
                        'lift_1':['mae'],
                        'drag_2':['mae'],
                        'lift_2':['mae']} )

def fit_model(model,
          lr_scheduler,
          early_stopping,
          checkpoint_weight,
          csv_logger,
          train,
          val):
    print("=================  Starting training ===================")
    start_time_for_fit = time.time()
    history = model.fit(train,epochs=config['epoch'],validation_data = val,
                        callbacks=[lr_scheduler,early_stopping,checkpoint_weight,csv_logger])
    end_time_for_fit = time.time()
    Training_time = end_time_for_fit - start_time_for_fit
    save_format(model,Training_time)
    print(f"TRAINING TIME IN  {Training_time:.4f} segundos.")
    return history

def evaluate (model,test):
    # 1. Evaluar el modelo obteniendo un diccionario
    results = model.evaluate(test, return_dict=True)
   
    # 2. Guardar en un archivo .txt (o .json)
    with open('resultados_evaluacion.txt', 'w') as f:
        json.dump(results, f, indent=4)

    print("Resultados de evaluacion guardados exitosamente.")

def predict_value(best_weights_path,M):
    M.load_weights(best_weights_path)
    pred_field_data,pred_force_data,true_data=prediction(M,test)
    return pred_field_data,pred_force_data,true_data
#  ========== Cambiar según modelo ==============================0000

def u_net_mlp():
    dir_save_weight='U_Net_Mlp'
    lr_scheduler,optimizer, early_stopping,checkpoint_weight,csv_logger=callbacks(dir_save_weight)
    model=unet_mlp_make_model()
    model.summary()
    compile_moldel(model,optimizer)
    history = fit_model(model,lr_scheduler,early_stopping,checkpoint_weight,csv_logger,train,val)
    return history,model

# Punto de entrada principal
if __name__ == "__main__":
    history,M=u_net_mlp()
    plot_loss(history)


K.clear_session()
    
    