from .node import Node
import numpy as np

class Constant(Node):
    def __init__(self, value:float = 1,  null = False,**kwargs):
        
        super().__init__(arity = 0, value = value)
        self.dW = 1 
        self.bias = 0
        self.null = null
    
    def __str__(self) -> str:
        return 'null' if self.null else '1'
    
    def __call__(self, X):
        #make shape consistent
        return np.ones(X.shape[0], dtype = np.float64)
        