from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper

@numba_operator_wrapper
def aq(operands: np.ndarray):
    r'''
    operands: np.ndarray of 1d np.array
    '''
    return operands[0] / np.sqrt(operands[1] ** 2 + 1)


class AQ(Node):
    def __init__(self,**kwargs):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '/'
    
    def __call__(self, operands: np.ndarray, update_stats= False):
        out =  aq(operands)
        self.dW = out
        self.dX = np.empty((2, operands.shape[1]), dtype = np.float64)
        self.dX[0] = self.value / np.sqrt(operands[1] ** 2 + 1)
        self.dX[1] = - self.value * operands[0] * operands[1] / (1 + operands[1] ** 2) ** 1.5
        
        assert self.dX.ndim == 2, self.dX.ndim
        if update_stats:
            self.update_stats(out)
        return out