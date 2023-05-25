from sympy import Expr, log, Abs, sign as Sign, Piecewise
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_with_grad_wrapper
from typing import List

@numba_operator_with_grad_wrapper
def nblog(x):
    x = np.ravel(x)
    margin = np.abs(x)
    margin = np.log(margin + 1)
    sign = np.sign(x)
    
    dX = sign / (margin + 1)
    return sign * margin,  np.expand_dims(dX, axis = 0)



class Log(Node):
    is_nonlinear = True
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'log'
    
    def __call__(self, X, **kwargs):
        out, self.dX = nblog(X)
        
        self.dW = out
        
        assert self.dX.ndim == 2, self.dX.ndim
        assert self.dX.dtype == np.float32
        
            

        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        margin = Abs(*X)
        
        _log = log(margin + 1)
        sign = Sign(*X)
        return _log * sign