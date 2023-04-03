from .node import Node
import numpy as np
# import numba as nb

# @nb.njit
def exp(X):
    return np.exp(X)

#deprecated
class Exp(Node):
    is_nonlinear = True
    def __init__(self):
        super().__init__(arity = 1)
    
    def __str__(self) -> str:
        return 'exp'
    
    def __call__(self, X):
        return exp(X[0])