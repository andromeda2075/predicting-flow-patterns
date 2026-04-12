
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
from config import load_config

config = load_config()

number_of_filters = [32,64,128,256,512]

type_padding = 'same'
f_activation = 'relu'
f_activation_last_flow='relu'
f_activation_last_forces='linear'


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

def unet_mlp_make_model():
    
    image_input = Input((config['h'], config['w'], config['c']))
    
    conv1,down_block1 = encoder (image_input, number_of_filters[0])
    conv2,down_block2 = encoder (down_block1 , number_of_filters[1])
    conv3,down_block3 = encoder (down_block2 , number_of_filters[2])
    conv4,down_block4 = encoder (down_block3 , number_of_filters[3])

    conv5 = conv_block(number_of_filters[4],down_block4)
   
    #===========Rama 1:  Decoder predicción de campos ============
    up6=decoder(conv5,conv4,number_of_filters[3],transpose='yes')
    up7=decoder(up6,conv3,number_of_filters[2],transpose='yes')
    up8=decoder(up7,conv2,number_of_filters[1],transpose='yes')
    up9=decoder(up8,conv1,number_of_filters[0],transpose='yes')

    flow_output= Conv2D(4, (1, 1), activation=f_activation_last_flow,name='flow_output',padding=type_padding)(up9)

    #===========Rama 2:  MLP PARA COEFICIENTES (Cl, Cd) ============
    # 1. Transformación a vector unidimensional 1x512 mediante convolución
    x_mlp = layers.Conv2D(512, 1, activation='relu')(conv5)
    x_mlp = layers.GlobalAveragePooling2D()(x_mlp)

     # 2. Capas Densas (MLP)
    x_mlp = layers.Dense(128, activation='relu')(x_mlp)
    x_mlp = layers.Dense(64, activation='relu')(x_mlp)

    drag_1= layers.Dense(1, activation=f_activation_last_forces, name="drag_1")(x_mlp)
    lift_1= layers.Dense(1, activation=f_activation_last_forces, name="lift_1")(x_mlp)

    drag_2= layers.Dense(1, activation=f_activation_last_forces, name="drag_2")(x_mlp)
    lift_2= layers.Dense(1, activation=f_activation_last_forces, name="lift_2")(x_mlp)

    model = models.Model(inputs=image_input,outputs=[flow_output,drag_1,lift_1,drag_2,lift_2])

    return model  