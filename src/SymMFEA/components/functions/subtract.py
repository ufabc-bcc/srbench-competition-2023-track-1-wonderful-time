from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper

@numba_operator_wrapper
def nbsubtract(operands):
    return operands[0] - np.sum(operands[1:], axis = 0)


class Subtract(Node):
    def __init__(self, arity: int, **kwargs):
        assert arity > 1
        super().__init__(arity = arity)
    
    def __str__(self) -> str:
        return '-'
    
    def __call__(self, operands: np.ndarray, update_stats= False):
        '''
        operands 2d array
        first axis is number of operands
        second axis is batch size
        '''
        out = nbsubtract(operands)
        self.dW = out
        
        
        self.dX = np.full((operands.shape[0], operands.shape[1]), -self.value, np.float64)
        self.dX[0] = -self.dX[0]
        
        assert self.dX.ndim == 2, self.dX.ndim
        if update_stats:
            self.update_stats(out)
        return out 
        