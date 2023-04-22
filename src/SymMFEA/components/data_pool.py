import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from ..utils import create_shared_np

class DataPool:
    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, stratify:bool = False):
        tmp = train_test_split(X, y, test_size=test_size, stratify= y if stratify else None)
        
        self.X_train, self.X_val, self.y_train, self.y_val = tuple([
            create_shared_np(mat.shape, mat) for mat in tmp
        ])
            
    
class DataView:
    def __init__(self, data_pool: DataPool, sample: float = 1):
        self.data_pool = data_pool
        self.index: np.ndarray = np.random.permutation(data_pool.y_train.shape[0])[:int(sample * data_pool.y_train.shape[0])]
        self.val_index: np.ndarray = np.random.permutation(data_pool.y_val.shape[0])[:int(sample * data_pool.y_val.shape[0])]
    
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
