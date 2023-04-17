from .progress_bar import *
from .visualize import draw_tree

import multiprocessing as mp
import numpy as np
import ctypes
from typing import Iterable

def create_shared_np(shape: Iterable[int], val= None, dtype= None):
    if isinstance(shape, int):
        shape = (shape, )
        
    if dtype is None:
        dtype = ctypes.c_double 
    
    num_blocks = 1
    for s in shape:
        num_blocks *= s
        
        
    shared_arr = mp.Array(dtype, num_blocks)
    
    array = np.frombuffer(shared_arr.get_obj(), dtype= np.dtype(dtype)).reshape(shape)
    
    if val is not None:
        array[:] = val
        
    
    return array