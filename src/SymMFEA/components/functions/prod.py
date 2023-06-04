from sympy import Expr
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_wrapper, multiply
from typing import List

@numba_operator_wrapper
def prod(operands): 
    return operands[0] * operands[1]



class Prod(Node):
    __slots__ = []
    def __init__(self, **kwargs):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return '*'
    
    def __call__(self, operands: np.ndarray, **kwargs):
        out =  prod(operands)
        self.dW = out
        
        
        self.dX = multiply(operands[::-1], self.value)
        
        assert self.dX.dtype == np.float32
        assert self.dX.ndim == 2, self.dX.ndim
        
        
            

        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        return X[0] * X[1]