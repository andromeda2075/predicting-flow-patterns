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
        g,(p,v,vx,vy)=self.__data_generation(batch_g,batch_p,batch_v,
                                             batch_vx,batch_vy)
        return g,(p,v,vx,vy)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self,batch_g,batch_p,batch_v,batch_vx,batch_vy):
        g = np.array([np.load(f) for f in batch_g])
        p = np.array([np.load(f) for f in batch_p])
        v = np.array([np.load(f) for f in batch_v])
        vx = np.array([np.load(f) for f in batch_vx])
        vy = np.array([np.load(f) for f in batch_vy])

        return g,(p,v,vx,vy)
    
















class NpyDataGenerator(tf.keras.utils.Sequence):
    """
    Generador adaptado para una única salida con múltiples canales.
    """
    def __init__(self, path_x, path_y, file_ids, batch_size=32, shuffle=True):
        self.path_x = path_x
        self.path_y = path_y
        self.file_ids = file_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.output_keys = sorted(self.path_y.keys()) 
        self.on_epoch_end()

    def __len__(self):
        """Devuelve el número de lotes por época."""
        return int(np.floor(len(self.file_ids) / self.batch_size))

    def __getitem__(self, index):
        """Genera un lote de datos."""
        batch_file_ids = self.file_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_file_ids)
        return X, y

    def on_epoch_end(self):
        """Mezcla los índices después de cada época."""
        if self.shuffle:
            np.random.shuffle(self.file_ids)

    # --- MÉTODO MODIFICADO ---
    def __data_generation(self, batch_file_ids):
        """Carga y devuelve un lote de datos con salida apilada (stacked)."""
        batch_x = []
        batch_y = [] # Ahora y es una sola lista

        for file_id in batch_file_ids:
            # Cargar entrada X (sin cambios)
            file_path_x = os.path.join(self.path_x, f"{file_id}.npy")
            batch_x.append(np.load(file_path_x))

            # --- CAMBIO PRINCIPAL: Cargar y apilar las salidas ---
            
            # 1. Cargar cada salida individualmente
            outputs_to_stack = []
            for key in self.output_keys: # Itera sobre ['p', 'u', 'v']
                file_path_y = os.path.join(self.path_y[key], f"{file_id}.npy")
                outputs_to_stack.append(np.load(file_path_y))
            
            # 2. Apilarlas en un solo array a lo largo del último eje (eje de canales)
            # Si cada array es (alto, ancho), el resultado será (alto, ancho, 3)
            stacked_y = np.stack(outputs_to_stack, axis=-1)
            
            # 3. Añadir el array apilado al lote de salida
            batch_y.append(stacked_y)

        # Convierte las listas en arrays de NumPy.
        # Ahora 'y' será un único tensor de lote.
        # Forma de X: (batch_size, alto, ancho, canales_entrada)
        # Forma de y: (batch_size, alto, ancho, 3)
        X = np.array(batch_x)
        y = np.array(batch_y)
        
        return X, y
