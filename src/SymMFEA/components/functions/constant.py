from sympy import Expr, Float
from .node import Node
import numpy as np
from typing import List
from ...utils.functional import ONE

class Constant(Node):
    __slots__ = ['null']
    def __init__(self, null = False,**kwargs):
        super().__init__(arity = 0)
        self.dW = ONE 
        self.null = null
    
    def __str__(self) -> str:
        return 'null' if self.null else '1'
    
    def expression(self, X: List[Expr]= None) -> Expr:
        return Float(1)
    
    def __call__(self, X, **kwargs):
        #make shape consistent
        out = np.ones(X.shape[0], dtype = np.float32)
        
            
        return out        