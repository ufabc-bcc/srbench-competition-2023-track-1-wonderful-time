from .node import Node
import numpy as np
from ...utils.functional import numba_operator_with_grad_wrapper, numba_v2v_float_wrapper


@numba_v2v_float_wrapper
def sigmoid(x):
    z = np.exp(np.sign(-x) * x)
    z = np.where(x < 0, z, 1) / (1 + z)
    return z

@numba_operator_with_grad_wrapper
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
        return '_ \_'
    
    def __call__(self, operands, update_stats: bool= False):
        out, self.dX =  valve(operands)
        
        
        self.dW = out
        self.dX = self.value * self.dX
        assert self.dX.ndim == 2, self.dX.ndim
        if update_stats:
            self.update_stats(out)
        return out