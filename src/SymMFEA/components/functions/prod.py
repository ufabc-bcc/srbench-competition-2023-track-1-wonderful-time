from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper

@numba_operator_wrapper
def prod(operands): 
    return operands[0] * operands[1]

class Prod(Node):
    def __init__(self, **kwargs):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '*'
    
    def __call__(self, operands: np.ndarray, update_stats= False):
        out =  prod(operands)
        self.dW = out
        
        
        self.dX = self.value * operands[::-1]
        assert self.dX.ndim == 2, self.dX.ndim
        if update_stats:
            self.update_stats(out)
        return out