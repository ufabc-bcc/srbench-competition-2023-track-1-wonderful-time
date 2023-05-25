from typing import List
from sympy import Expr, Piecewise
from .node import Node
import numpy as np
import numba as nb
from ...utils.functional import numba_operator_wrapper, ONE, ZERO, numba_v2v_float_wrapper
import os 

@numba_operator_wrapper
def relu(X):
    return np.maximum(np.ravel(X), ZERO)

@nb.njit(nb.float32[:, :](nb.float32[:, :]), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None, nogil=True)
def calculate_dx(operands):
    return np.where(operands > ZERO, ONE, ZERO)


class Relu(Node):
    is_nonlinear = True
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'relu'
    
    def __call__(self, operands: np.ndarray, **kwargs):
        out = relu(operands)
        
        #calculate d
        self.dW = out
        self.dX = calculate_dx(operands)
        
        assert self.dX.dtype == np.float32
        assert self.dX.ndim == 2, self.dX.ndim

        
            
        
        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        x = X[0]
        return Piecewise((x, x > 0), (0, True))
    
    
    
    