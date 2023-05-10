import numpy as np
from typing import Tuple
from ..utils import create_shared_np, stratify_train_test_split

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
        
        self.index: np.ndarray = stratify_train_test_split(None, data_pool.y_train, test_size= sample, return_idx= True)
        self.val_index: np.ndarray = stratify_train_test_split(None, data_pool.y_val, test_size= sample, return_idx= True)
    
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

#so far dataloader use for train only
class TrainDataLoader:
    def __init__(self, data_view:DataView, batch_size:int = 10, shuffle: bool = True) -> None:
        self.data_view = data_view
        self.batch_size = batch_size
        self.i = 0
        self.shuffle = shuffle
        self.index: np.ndarray
        if shuffle:
            self.shuffle_view()
        else:
            self.index = np.arange(self.data_view.len_train)
            
            
    def unlock(self):
        self.data_view.unlock()
        
    @property
    def hasNext(self):
        has =  self.i < self.data_view.len_train
        if not has:
            self.i = 0
            if self.shuffle:
                self.shuffle_view()
        
        return has
        
    def __next__(self) -> Tuple[np.ndarray]:
        end  = min(self.i + self.batch_size, self.data_view.len_train)
        index = self.index[self.i : end]
        data = self.data_view.X_train[index], self.data_view.y_train[index]
        self.i += self.batch_size
        return data
    
    def shuffle_view(self):
        self.index = np.random.permutation(self.data_view.len_train)
        
def initDataPool(*args, **kwargs):
    global data_pool 
    data_pool = DataPool(*args, **kwargs)
