from .node import Node
import numpy as np
import numba as nb

@nb.njit
def nbsum(operands):
    return np.sum(operands, axis = 0)


class Sum(Node):
    def __init__(self, arity: int, **kwargs):
        assert arity > 1
        super().__init__(arity = arity)
    
    def __str__(self) -> str:
        return '+'
    
    def __call__(self, operands: np.ndarray):
        out =  nbsum(operands)
        self.dW = out
        self.dX = np.full((operands.shape[0], operands.shape[1]), self.value, dtype= np.float64)
        assert self.dX.ndim == 2, self.dX.ndim
        return out