from typing import List
from sympy import Expr, tanh
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper

@numba_operator_wrapper
def nbtanh(X):
    return np.tanh(np.ravel(X))

class Tanh(Node):
    is_nonlinear = True
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'tanh'
    
    def __call__(self, X):
        out = nbtanh(X)
        
        self.dW = out
        self.dX = np.expand_dims(1 - out ** 2, axis = 0)
        assert self.dX.ndim == 2, self.dX.ndim

        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        return tanh(*X)