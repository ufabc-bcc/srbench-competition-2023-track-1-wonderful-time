from sympy import Expr, Symbol
from .node import Node
from typing import List

class Operand(Node):
    def __init__(self, index: int = 0, value:float = 1, **kwargs):      
        super().__init__(arity = 0, value = value)
        assert index >= 0
        self.index = index
        
    
    def __str__(self) -> str:
        return 'x_{}'.format(self.index)
    
    def __call__(self, X, **kwargs):
        r'''
        X: 2d array
        '''
        out = X[:, self.index]
        self.dW = out
        
        
            

        return out
    
    
    def expression(self, X: List[Expr]= None) -> Expr:
        return Symbol(f"x{self.index}", real= True)