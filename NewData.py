import numpy as np
from sklearn.model_selection import train_test_split

def load_data(inp_path,out_path):
    # 0: Re, 1: SDF 2:Mask
    inp=np.load(inp_path)
    out=np.load(out_path)
    X=inp['data'] # [batch,channel,h,w]
    Y=out['data']
    XT = np.transpose(X, (0,2, 3, 1))
    YT=np.transpose(Y, (0,2, 3, 1))
    print(f'X input shape {XT.shape}')
    print(f'Y output shape {YT.shape}')
    #Preprocessing
    XT_SDF=XT[:,:,:,1]/np.max(XT[:,:,:,1])
    XT_bin=XT[:,:,:,2] / 255. 
    XT_Re=XT[:,:,:,0]/np.max(XT[:,:,:,0])
    X_in = np.stack([XT_Re, XT_SDF, XT_bin], axis=-1)
    return X_in,YT

def split_data(inp_path,out_path,ntrain,ntest):
    x,y=load_data(inp_path,out_path)
    X_train, X_temp, Y_train, Y_temp = train_test_split(x, y, test_size=ntrain)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=ntest)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train,X_val,X_test,Y_train,Y_val,Y_test 

