from sympy import Expr, sin
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_with_grad_wrapper
from typing import List

@numba_operator_with_grad_wrapper
def nbsin(x):
    x = np.ravel(x)
    
    return np.sin(x),  np.expand_dims(np.cos(x), axis = 0)



class Sin(Node):
    is_nonlinear = True
    __slots__ = []
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'sin'
    
    def __call__(self, X, **kwargs):
        out, self.dX = nbsin(X)
        
        self.dW = out
        
        assert self.dX.ndim == 2, self.dX.ndim
        assert self.dX.dtype == np.float32
        
            

        return out
    
    def expression(self, X: List[Expr]) -> Expr:

        return sin(X[0])