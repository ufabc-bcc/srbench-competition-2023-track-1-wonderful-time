from .node import Node
import numpy as np
# import numba as nb

# @nb.njit
def tanh(X):
    return np.tanh(X)

class Tanh(Node):
    is_nonlinear = True
    def __init__(self):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'tanh'
    
    def __call__(self, X):
        out = tanh(X[0])
        
        self.dW = out
        self.dX = [1 - out ** 2]
        
        return out