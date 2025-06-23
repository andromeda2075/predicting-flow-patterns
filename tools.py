#Generic imports
from sklearn.metrics                    import mean_squared_error,r2_score,mean_absolute_error

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import random


def predicting_error(dataset,model):
    print('*** Starting ***')
    xtrue=[]
    ytrue=[]
    ypred=[]
    error_abs=[]
    start_time_pred=time.time() 
    for k in range(len(dataset)):  # k=1,2,...,N (N: total batches)
        x,y=dataset[k]
        ytrue.append(y)
        y_preds=model.predict(x)
        ypred.append(y_preds)
    end_time_pred=time.time() 
    inference_time = end_time_pred - start_time_pred
    print(f"PREDICTING TIME IN  {inference_time:.4f} seconds.")
    return ytrue,ypred

def relative_error(yp,yt):
    err_rel=list()
    for n_batch in range(len(yt)):
        for j in range(yt[0].shape[0]):
            e=np.abs(yp[n_batch][j]-yt[n_batch][j])
            err=np.mean(e/(np.abs(yt[n_batch][j]) + 1e-4),axis=(0,1))
            err_rel.append(err)
    error_relativos = np.array(err_rel)
    df = pd.DataFrame(error_relativos, columns=["P", "V", "VX","VY"])
    # Boxplot
    sns.boxplot(data=df)
    plt.ylabel("Relative Error")
    plt.title("Distribution of relative error by channel")
    plt.savefig("Relative_error_boxplot.png", dpi=300, bbox_inches='tight')  # alta resolución
    plt.show()

def metric_by_channels(ypred,ytrue):
    y_preds = np.concatenate(ypred, axis=0)  # (N, H, W, 4)
    y_trues = np.concatenate(ytrue, axis=0)
    
    # MSE, MAE, RMSE by channel
    for i, var in enumerate(["p", "v", "vx", "vy"]):
        mse = mean_squared_error(y_trues[..., i].ravel(), y_preds[..., i].ravel())
        mae = mean_absolute_error(y_trues[..., i].ravel(), y_preds[..., i].ravel())
        rmse = np.sqrt(mse)
        print(f"{var}: MSE = {mse:.6f}, MAE = {mae:.6f}, RMSE ={rmse:.6f}")
    
    
    y_true_flat = y_trues.flatten()
    y_pred_flat = y_preds.flatten()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    maepercent=mae*100
    print(f"MSE global: {mse:.6f}")
    print(f"MAE global: {mae:.6f}")
    print(f"MAE % global: {maepercent:.6f}")
    print(f"RMSE global: {rmse:.6f}")

def plot_model(history,name1,name2,title,option=True):
    titlee=title+'semilog'
    pd.DataFrame(history.history)[[name1,name2]].plot(figsize=(8, 5))
    plt.yscale('log')       # Escala logarítmica para el eje Y
    plt.grid(True, which='both')  # Mostrar grid tanto en escala mayor como menor
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(f'{name1} (log)')
    plt.tight_layout()
    plt.savefig(titlee, dpi=300, bbox_inches='tight')  # alta resolución
    plt.show()
    if option:
        pd.DataFrame(history.history)[[name1,name2]].plot(figsize=(8, 5))
        plt.grid(True, which='both')  # Mostrar grid tanto en escala mayor como menor
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(f'{name1}')
        plt.tight_layout()
        plt.savefig(title, dpi=300, bbox_inches='tight')  # alta resolución
        plt.show()
            


def plotting(ytrue,ypred,name='visualization'):
    n = random.randint(0, len(ytrue)-1)
    m = random.randint(0, ytrue[0].shape[0]-1)
    titles = [ 'P pred', 'P true','V pred', 'V true','Vx pred', 'Vx true','Vy pred', 'Vy true']
    fig, axes = plt.subplots(4, 3, figsize=(16, 10))
    error=[]
    
    for i in range(4):
        error.append(np.abs(ytrue[n][m,:,:,i]- ypred[n][m,:,:,i]))
        axes[i,0].imshow(ypred[n][m,:,:,i], cmap='gray')
        axes[i, 0].set_title(titles[2*i])
        axes[i, 0].axis('off')

        axes[i,1].imshow(ytrue[n][m,:,:,i], cmap='gray')
        axes[i, 1].set_title(titles[2*i+1])
        axes[i, 1].axis('off')

        e=axes[i,2].imshow(error[0], cmap='turbo')
        axes[i, 2].set_title('Absolute Error')
        axes[i, 2].axis('off')
        plt.colorbar(e, ax=axes[i,2], fraction=0.046)
    fig.savefig(name, dpi=300)
                    

#Optional 
def MSEplus(u_true, u_pred):
    
  Eps=1e-4
  
  Ip_value = tf.reduce_sum(tf.cast(u_true != 0.0, tf.float64)) 
  u_true = tf.cast(u_true, dtype=tf.float64)
  u_pred = tf.cast(u_pred, dtype=tf.float64)  
  E_normL2=tf.sqrt(tf.reduce_sum(tf.square(u_true - u_pred)))
  E=tf.reduce_sum(tf.square(u_true - u_pred))
  u_pred_normL2=tf.sqrt(tf.reduce_sum(tf.square(u_pred)))
  mse_loss = tf.reduce_sum(E + E_normL2 / (u_pred_normL2 + Eps))
  mse_loss=mse_loss/Ip_value
 
  return mse_loss 
        
       
    
    
