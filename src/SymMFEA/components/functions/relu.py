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
    
    def __call__(self, operands: np.ndarray, update_stats= False):
        out = relu(operands)
        
        #calculate d
        self.dW = out
        self.dX =  np.where(operands > 0, 1.0, 0.0)
        assert self.dX.ndim == 2, self.dX.ndim
        if update_stats:
            self.update_stats(out)
        return out
    
    
    
    
    