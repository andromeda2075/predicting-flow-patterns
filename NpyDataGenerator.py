import numpy as np
import tensorflow as tf
import os
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#https://satvik-venkatesh.github.io/blog-posts/data-gen-keras.html

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,path_g,path_p,path_v,path_vx,path_vy, batch_size = 10,shuffle = False):
        self.path_g = path_g
        self.path_p= path_p
        self.path_v= path_v
        self.path_vx=path_vx
        self.path_vy=path_vy
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.indexes = np.arange(len(self.path_g))
    
    def __len__(self):
        return int(np.floor(len(self.path_g) / self.batch_size))
    
    def __getitem__(self,index):
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        batch_g=[self.path_g[k] for k in  batch_indexes]                         
        batch_p=[self.path_p[k] for k in  batch_indexes]
        batch_v=[self.path_v[k] for k in  batch_indexes]
        batch_vx=[self.path_vx[k] for k in  batch_indexes]
        batch_vy=[self.path_vy[k] for k in  batch_indexes]
        g,y=self.__data_generation(batch_g,batch_p,batch_v,
                                             batch_vx,batch_vy)
        return g,y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self,batch_g,batch_p,batch_v,batch_vx,batch_vy):
        g = np.array([np.load(f) for f in batch_g])
        p = np.array([np.load(f) for f in batch_p])
        v = np.array([np.load(f) for f in batch_v])
        vx = np.array([np.load(f) for f in batch_vx])
        vy = np.array([np.load(f) for f in batch_vy])
        y = np.concatenate([p,v,vx,vy], axis=-1)
        return g,y



