from .node import Node
import numpy as np
# import numba as nb

# @nb.njit
def sum(operands: list):
    r'''
    operands: list of 2d np.array
    '''
    return operands[0] + operands[1]


class Sum(Node):
    def __init__(self):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '+'
    
    def __call__(self, operands: list):
        out =  sum(operands)
        self.dW = out
        self.dX = [self.value, self.value]
        #normalize to prevent overflow
        # self.dX = [1, 1]
        return out