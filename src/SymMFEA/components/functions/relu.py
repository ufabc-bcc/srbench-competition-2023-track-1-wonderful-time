from .node import Node
import numpy as np
# import numba as nb

# @nb.njit
def relu(X):
    return np.maximum(X, 0)

class Relu(Node):
    is_nonlinear = True
    def __init__(self):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'relu'
    
    def __call__(self, operands: list):
        out = relu(operands[0])
        
        #calculate d
        self.dW = out
        self.dX =  [np.maximum(operands[0], 0) * self.value]
        
        return out
    
    
    
    
    