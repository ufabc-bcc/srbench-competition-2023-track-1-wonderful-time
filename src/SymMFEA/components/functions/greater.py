from sympy import Expr, Piecewise
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper, ONE
from typing import List

@numba_operator_wrapper
def nb_greater(x):
    return np.where(x[0] > x[1], ONE, np.float32(0.0))



class Greater(Node):
    is_nonlinear = True
    __slots__ = []
    def __init__(self, **kwargs):
        super().__init__(arity = 2)
        
    
    def __str__(self) -> str:
        return '>'
    
    def __call__(self, X, **kwargs):
        out = nb_greater(X)
        
        self.dW = out
        self.dX = np.zeros_like(X, dtype = np.float32)
        
        
        assert self.dX.ndim == 2, self.dX.ndim
        assert self.dX.dtype == np.float32
        
            

        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        return Piecewise((1, X[0] > X[1]), (0, True))