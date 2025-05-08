'''
cat /proc/cpuinfo | grep "model name" | uniq
model name	: 13th Gen Intel(R) Core(TM) i9-13900HX
'''
# Cores
n_threads= 18 # Max 32

# Paths
save_in='/home/ppgi/Trabajo/Codigos_generate_data/DLCODES-VER-5'
open_dir = "/home/ppgi/Trabajo/Codigos_generate_data/DLCODES-VER-5/"
file=['Geometry','Magnitude','Pression','U001','U002']

Geo=open_dir+file[0]
Mag=open_dir+file[1]
Pr=open_dir+file[2]
Vx=open_dir+file[3]
Vy=open_dir+file[4]

list_paths=[Geo,Mag,Pr,Vx,Vy]

#Dataset split 
n_train=0.8   
n_valid=0.   
n_test=0.2

# number of total data
n_sample= 20000

# Pack number
pack_size              =100

# Model name
model="U-Net"
plot_name='U-Net_Model.png'

# Image data
img_width             =  256   # 739   G:737
img_height            =  128   # 185
channel               =  1
#opt_for_data          =  1     # 1-Mag  2-Pr  3-Vx  4-Vy  'all'- all data 
#opt_process           =  False # True: binary  False: Gray

# Hiperparameters
num_classes=4
num_epochs =300
batch_size=100
patience=150   # How long to wait after last time validation loss improved
LR=0.0001                              



'''
# Network data
n_filters_initial       = 64
kernel_size             = 3
kernel_transpose_size   = 2
stride_size_pool        = 2  #1
padding                 = 'same' # 'same' keeping size  /  'valid' (unpadding)
activation_function     = 'relu'

#Parameters
Loss                    ='mean_squared_error'
metric                  ='mean_absolute_error'
lr                      =1e-4
batch_size              = 20
num_epochs              = 200

'''
