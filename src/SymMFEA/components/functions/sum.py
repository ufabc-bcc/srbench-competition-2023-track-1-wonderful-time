from typing import List
from sympy import Expr, Add
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper

@numba_operator_wrapper
def nbsum(operands):
    return np.sum(operands, axis = 0)


class Sum(Node):
    def __init__(self, arity: int, **kwargs):
        assert arity > 1
        super().__init__(arity = arity)
    
    def __str__(self) -> str:
        return '+'
    
    def __call__(self, operands: np.ndarray, update_stats= False, **kwargs):
        out =  nbsum(operands)
        self.dW = out
        self.dX = np.full((operands.shape[0], operands.shape[1]), self.value, dtype= np.float64)
        assert self.dX.ndim == 2, self.dX.ndim
        
        if update_stats:
            self.update_stats(out)

        return out
    
    
    def expression(self, X: List[Expr]) -> Expr:
        return Add(*X)