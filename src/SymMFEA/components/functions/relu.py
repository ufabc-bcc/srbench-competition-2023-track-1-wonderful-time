from typing import List
from sympy import Expr, Piecewise, Float
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper

@numba_operator_wrapper
def relu(X):
    return np.maximum(np.ravel(X), 0)

class Relu(Node):
    is_nonlinear = True
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'relu'
    
    def __call__(self, operands: np.ndarray):
        out = relu(operands)
        
        #calculate d
        self.dW = out
        self.dX =  np.where(operands > 0, 1.0, 0.0)
        assert self.dX.ndim == 2, self.dX.ndim

        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        x = X[0]
        return Piecewise((x, x > 0), (True, 0))
    
    
    
    