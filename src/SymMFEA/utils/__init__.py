from .progress_bar import *
from .visualize import draw_tree
import multiprocessing as mp
import numpy as np
import ctypes
from typing import Iterable, Union
from sympy import Expr
from k_means_constrained import KMeansConstrained
from sklearn.model_selection import train_test_split
from queue import Full
QUEUE_SIZE = 1000000000

def create_shared_np(shape: Iterable[int], val: Union[float, np.array]= 0.0, dtype= None, race_safe= False):
    if isinstance(shape, int):
        shape = (shape, )
        
    if dtype is None:
        dtype = ctypes.c_float 
    
    num_blocks = 1
    for s in shape:
        num_blocks *= s
        
    if race_safe:
        shared_arr = mp.Array(dtype, num_blocks)
        
        array = np.frombuffer(shared_arr.get_obj(), dtype= np.dtype(dtype)).reshape(shape)
    else:
        shared_arr = mp.RawArray(dtype, num_blocks)
        
        array = np.frombuffer(shared_arr, dtype= np.dtype(dtype)).reshape(shape)
        
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
    discrete_y = kmean(np.copy(y).reshape(-1, 1), n_init= 1, n_clusters= 8, size_min = 10, random_state= 0)
    
    test_size = float(test_size)
    if return_idx:
        if test_size == 1:
            return np.arange(y.shape[0])
        return train_test_split(np.arange(y.shape[0]), y, test_size=test_size, stratify= discrete_y)[1]
    else:
        return train_test_split(X, y, test_size=test_size, stratify= discrete_y)
    
#put to queue
def _put(jobs, inqueue):
    is_put = False
    while not is_put:
        try:
            inqueue.put_many(jobs)
        except Full:
            ...
        else:
            is_put = True
            
class Worker:
    def __init__(self, func, *args, **kwargs):
        
        self.process = mp.Process(target= func, args=args, kwargs= kwargs)
        
        self.process.start()
        
        
    def kill(self):
        self.process.join()
        self.process.terminate()