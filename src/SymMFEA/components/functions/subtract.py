from .node import Node
import numpy as np
import numba as nb

@nb.njit
def nbsubtract(operands):
    return operands[0] - np.sum(operands[1:], axis = 0)


class Subtract(Node):
    def __init__(self):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '-'
    
    def __call__(self, operands: np.ndarray):
        '''
        operands 2d array
        first axis is number of operands
        second axis is batch size
        '''
        out = nbsubtract(operands)
        self.dW = out
        
        
        self.dX = np.full((operands.shape[0], operands.shape[1]), -self.value, np.float64)
        self.dX[0] = -self.dX[0]
        
        assert self.dX.ndim == 2
        return out 
        