from typing import List
from sympy import Expr, sign as Sign, Piecewise, exp as Exp
from .node import Node
import numpy as np
from ...utils.functional import numba_operator_with_grad_wrapper, numba_v2v_float_wrapper, ONE


@numba_v2v_float_wrapper
def sigmoid(x):
    z = np.exp(np.sign(-x) * x)
    z = np.where(x < 0, z, ONE) / (ONE + z)
    return z

@numba_operator_with_grad_wrapper
def valve(operands):
    
    switch = sigmoid(operands[0])
    out = switch * operands[1]
    
    dX = np.empty((2, operands.shape[1]), dtype = np.float32)
    dX[0] = out * (1 - switch)
    dX[1] = switch
    return out, dX


class Valve(Node):
    def __init__(self, **kwargs):
        super().__init__(arity = 2)
    
    def __str__(self) -> str:
        return 'valve'
    
    def __call__(self, operands, **kwargs):
        out, self.dX =  valve(operands)
        
        
        self.dW = out
        self.dX = self.value * self.dX
        assert self.dX.ndim == 2, self.dX.ndim
        
        
            

        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        x0 = X[0]
        z = Exp(-Sign(x0) * x0)
        switch = Piecewise((z, x0 < 0), (1, True)) / (1 + z)
        return switch * X[1]