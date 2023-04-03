from .node import Node
import numpy as np
from ...utils.functional import normalize
# import numba as nb

# @nb.njit
def aq(operands: list):
    r'''
    operands: list of 1d np.array
    '''
    return operands[0] / np.sqrt(operands[1] ** 2 + 1)


class AQ(Node):
    def __init__(self):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '/'
    
    def __call__(self, operands: list):
        out =  aq(operands)
        self.dW = out
        
        
        
        self.dX = [self.value / np.sqrt(operands[1] ** 2 + 1),
                   - self.value * operands[0] * operands[1] / (1 + operands[1] ** 2) ** 1.5]
        
        return out