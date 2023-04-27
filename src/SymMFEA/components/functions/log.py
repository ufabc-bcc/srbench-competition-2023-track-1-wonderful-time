from .node import Node
import numpy as np
from ...utils.functional import numba_operator_with_grad_wrapper

@numba_operator_with_grad_wrapper
def log(x):
    x = np.ravel(x)
    margin = np.abs(x)
    margin = np.where(margin > 1, 1 + np.log(margin), margin)
    sign = np.sign(x)
    
    dX = np.where(margin > 1, 1 / margin, 1)
    return sign * margin,  np.expand_dims(dX * sign, axis = 0)



class Log(Node):
    is_nonlinear = True
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'log'
    
    def __call__(self, X):
        out, self.dX = log(X)
        
        self.dW = out
        
        assert self.dX.ndim == 2, self.dX.ndim

        return out