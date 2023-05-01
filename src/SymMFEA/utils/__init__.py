from .progress_bar import *
from .visualize import draw_tree

import multiprocessing as mp
import numpy as np
import ctypes
from typing import Iterable, Union
from sympy import Expr

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


def handle_number_of_list(param: Union[int, float, list], size: int) -> list:
    return param if isinstance(param, list) else [param] * size


def count_nodes(expr: Expr):
    '''
    Counts the nodes of a sympy expression.

    Parameters
    ----------
    expr : sympy
                sympy expression as created by SymExpr class.
    '''
    count = 0
    for arg in expr.args:
        count += count_nodes(arg)
    return count + 1
