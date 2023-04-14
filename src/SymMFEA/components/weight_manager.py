import multiprocessing as mp
import numpy as np
import ctypes
from typing import Iterable

#NOTE: so far not remove not used memory

def create_shared_np(shape: Iterable[int]):
    num_blocks = 1
    for s in shape:
        num_blocks *= s
    shared_arr = mp.Array(ctypes.c_double, num_blocks)
    return np.frombuffer(shared_arr.get_obj()).reshape(shape)


class WeightManager:
    def __init__(self, shape: tuple= (1000000, 50)):
        self.index = 0    
        self.len = shape[0]
        self.weight = create_shared_np(shape)
        self.bias = create_shared_np(shape)
        self.best_weight = create_shared_np(shape)
        self.best_bias = create_shared_np(shape)
        self.dW = create_shared_np(shape)
        self.dB = create_shared_np(shape)
        
    
    def __next__(self):
        tmp = self.index
        self.index += 1
        return tmp

def initWM(shape):
    global WM
    WM = WeightManager(shape)
