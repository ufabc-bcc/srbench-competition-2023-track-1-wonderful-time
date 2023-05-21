import numpy as np
from typing import Tuple
from ..utils import create_shared_np, stratify_train_test_split
import ctypes

class DataPool:
    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, **kwargs):
        
        tmp = stratify_train_test_split(X, y, test_size=test_size, **kwargs)
        
        self.X_train, self.X_val, self.y_train, self.y_val = tuple([
            create_shared_np(mat.shape, mat) for mat in tmp
        ])
        
    @property
    def train_size(self):
        return self.X_train.shape[0]
    
    @property
    def val_size(self):
        return self.X_val.shape[0]
    
    
    
class DataView:
    def __init__(self, data_pool: DataPool, sample: float = 1):
        self.data_pool = data_pool
        
        idx = stratify_train_test_split(None, data_pool.y_train, test_size= sample, return_idx= True)
        self.index = create_shared_np((len(idx),), val = idx, dtype= ctypes.c_ulong)
        
        idx = stratify_train_test_split(None, data_pool.y_val, test_size= sample, return_idx= True)
        self.val_index = create_shared_np((len(idx),), val = idx, dtype= ctypes.c_ulong)
        
    @property
    def len_train(self) -> int:
        return self.y_train.shape[0]
    
    @property
    def X_train(self) ->np.ndarray:
        return self.data_pool.X_train[self.index]
    
    @property
    def y_train(self) ->np.ndarray:
        return self.data_pool.y_train[self.index]
    
    @property
    def X_val(self) ->np.ndarray:
        return self.data_pool.X_val[self.val_index]
    
    @property
    def y_val(self) ->np.ndarray:
        return self.data_pool.y_val[self.val_index]
    
    def unlock(self):
        self.index = np.arange(self.data_pool.train_size)
        self.val_index = np.arange(self.data_pool.val_size)

        
def initDataPool(*args, **kwargs):
    global data_pool 
    data_pool = DataPool(*args, **kwargs)
