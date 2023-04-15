from .progress_bar import *
from .visualize import draw_tree

import multiprocessing as mp
import numpy as np
import ctypes
from typing import Iterable

def create_shared_np(shape: Iterable[int], val= None):
    num_blocks = 1
    for s in shape:
        num_blocks *= s
    shared_arr = mp.Array(ctypes.c_double, num_blocks)
    
    array = np.frombuffer(shared_arr.get_obj()).reshape(shape)
    if val is None:
        return array
    else:
        array[:] = val[:]
    
    return array