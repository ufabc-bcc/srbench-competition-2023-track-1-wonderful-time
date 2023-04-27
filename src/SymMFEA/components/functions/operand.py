from .node import Node
import numpy as np

class Operand(Node):
    def __init__(self, index: int = 0, value:float = 1, bias = 0, **kwargs):      
        super().__init__(arity = 0, value = value, bias = bias)
        assert index >= 0
        self.index = index
        
    
    def __str__(self) -> str:
        return 'x_{}'.format(self.index)
    
    def __call__(self, X):
        r'''
        X: 2d array
        '''
        out = X[:, self.index]
        self.dW = out

        return out