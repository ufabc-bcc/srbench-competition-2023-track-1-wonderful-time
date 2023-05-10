from .progress_bar import *
from .visualize import draw_tree

import multiprocessing as mp
import numpy as np
import ctypes
from typing import Iterable, Union
from sympy import Expr
from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split

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


def kmean(y: np.ndarray, **kwargs):
    model = KMeansConstrained(**kwargs)
    return model.fit_predict(y)

def stratify_train_test_split(X:np.ndarray, y: np.ndarray, test_size: float, return_idx:bool= False, **kwargs):
    discrete_y = kmean(y.reshape(-1, 1), n_init= 1, n_clusters= 8, size_min = 10, random_state= 0)
    
    test_size = float(test_size)
    if return_idx:
        if test_size == 1:
            return np.arange(y.shape[0])
        return train_test_split(np.arange(y.shape[0]), y, test_size=test_size, stratify= discrete_y)[1]
    else:
        return train_test_split(X, y, test_size=test_size, stratify= discrete_y)