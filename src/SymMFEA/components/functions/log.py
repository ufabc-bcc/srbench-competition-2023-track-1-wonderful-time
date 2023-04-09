from .node import Node
import numpy as np
from numba import jit

@jit(nopython = True)
def log(x):
    margin = np.abs(x)
    margin = np.where(margin > 1, 1 + np.log(margin), margin)
    sign = np.sign(x)
    
    dX = np.where(margin > 1, 1 / margin, 1)
    return sign * margin, dX * sign



class Log(Node):
    is_nonlinear = True
    def __init__(self, **kwargs):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'log'
    
    def __call__(self, X):
        out, dX = log(X[0])
        
        self.dW = out
        self.dX = np.expand_dims(dX, axis= 0)
        assert self.dX.ndim == 2, self.dX.ndim
        return out