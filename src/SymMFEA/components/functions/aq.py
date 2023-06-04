from typing import List
from sympy import Expr, sqrt
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper, ONE
import numba as nb
import os 

@numba_operator_wrapper
def aq(operands: np.ndarray):
    r'''
    operands: np.ndarray of 1d np.array
    '''
    return operands[0] / np.sqrt(operands[1] ** 2 + ONE)

@nb.njit(nb.float32[:, :](nb.float32[:, :], nb.float32), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None, nogil=True)
def calculate_dx(operands, value):
    dx = np.empty((2, operands.shape[1]), dtype = np.float32)
    dx[0] = value / np.sqrt(operands[1] ** 2 + 1)
    dx[1] = - value * operands[0] * operands[1] / (ONE + operands[1] ** 2) ** 1.5
    return dx


class AQ(Node):
    __slots__ = []
    def __init__(self,**kwargs):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '/'
    
    def __call__(self, operands: np.ndarray, **kwargs):
        out =  aq(operands)
        self.dW = out
        self.dX = calculate_dx(operands, self.value)
        
        
        assert self.dX.ndim == 2, self.dX.ndim
        assert self.dX.dtype == np.float32
        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        return X[0] / sqrt(X[1] ** 2 + 1)