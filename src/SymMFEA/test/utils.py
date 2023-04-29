
import numpy as np
from typing import Tuple, List
def generate_input_list(shape: Tuple[int], size= 10):
    return [ 
         np.random.rand(*shape) for _ in range(size)
         ]

def is_closed(x1: np.ndarray, x2: np.ndarray):
    return np.allclose(x1, x2, rtol= 1e-5, atol= 1e-5)

def zip_inputs(*ls: List[list]) -> list:
    ls = list(ls)
    for l in ls:
        if isinstance(l,list):
            common_lens = len(l)
    
    for i in range(len(ls)):
        if not isinstance(ls[i], list):
            ls[i] = [l] * common_lens
    
    if len(ls) > 1:
        return [
            tuple([ls[i][j] for i in range(len(ls))]) for j in range(len(ls[0]))
        ]
    
    else:
        return ls
    
