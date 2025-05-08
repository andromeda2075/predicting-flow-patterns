'''
cat /proc/cpuinfo | grep "model name" | uniq
model name	: 13th Gen Intel(R) Core(TM) i9-13900HX
'''

# Paths

save_in='/home/ppgi/Trabajo/Codigos_generate_data/DLCODES-VER-5'
open_dir = "/home/ppgi/Trabajo/Codigos_generate_data/DLCODES-VER-5/"

#save_in='/home/guiomar/Desktop/CODES/DLCODES-VER-5'
#open_dir = "/home/guiomar/Desktop/CODES/Generation_Data/Proyecto_UNI/"

file=['Geometry','Magnitude','Pression','U001','U002']

Geo=open_dir+file[0]
Mag=open_dir+file[1]
Pr=open_dir+file[2]
Vx=open_dir+file[3]
Vy=open_dir+file[4]

list_paths=[Geo,Mag,Pr,Vx,Vy]

#Dataset split 
n_split=0.2
n_batch = 500

# number of total data
n_sample= 20000

# Model name
model="U-Net"
plot_name='U-Net_Model.png'

# Image data
img_width             =  256   # 739   G:737
img_height            =  128   # 185
channel               =  1

# Hiperparameters
num_classes=4
num_epochs =300
batch_size=100
patience=100  # How long to wait after last time validation loss improved
LR=0.0001                    
