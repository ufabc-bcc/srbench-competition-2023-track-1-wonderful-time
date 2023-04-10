from .node import Node
import numpy as np
import numba as nb

@nb.njit(nb.float64[:](nb.float64[:]))
def sigmoid(x):
    z = np.exp(np.sign(-x) * x)
    z = np.where(x < 0, z, 1) / (1 + z)
    return z

@nb.njit
def valve(operands):
    
    switch = sigmoid(operands[0])
    out = switch * operands[1]
    
    dX = np.empty((2, operands.shape[1]), dtype = np.float64)
    dX[0] = out * (1 - switch)
    dX[1] = switch
    return out, dX


class Valve(Node):
    def __init__(self, **kwargs):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '*'
    
    def __call__(self, operands: np.ndarray):
        out, self.dX =  valve(operands)
        
        
        self.dW = out
        self.dX = self.value * self.dX
        assert self.dX.ndim == 2, self.dX.ndim
        return out