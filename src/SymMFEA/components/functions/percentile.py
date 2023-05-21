# from sympy import Expr, Symbol, Piecewise
# from .node import Node
# from typing import List
# import random
# import numpy as np
# import numba as nb
# import os
# @nb.njit(nb.types.Tuple((nb.float64[:], nb.float64))(nb.float64[:,:], nb.float64, nb.int64), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
# def training_percentile(X, p, index):
#     out = X[:, index]
#     threshold = np.percentile(out, p)
#     out = np.where(out > threshold, 1.0, 0.0)
#     return out, threshold
    


# class Percentile(Node):
#     percentiles: List[float] = [float(i) for i in range(5, 100, 5)]
    
    
#     def __init__(self, index: int = 0, value:float = 1, **kwargs):      
#         super().__init__(arity = 0, value = value)
#         assert index >= 0
#         self.index = index
#         self.attrs['p'] = self.percentiles[random.randint(0, len(self.percentiles) - 1)]
        
#     def __str__(self) -> str:
#         return '%x{}'.format(self.index)
    
#     def __call__(self, X, training = False, **kwargs):
#         r'''
#         X: 2d array
#         '''
#         if training:
#             assert self.attrs['p'] > 0 and self.attrs['p'] < 100
#             out, threshold = training_percentile(X, self.attrs['p'], self.index)
#             if self.attrs.get('threshold') is None:
#                 self.attrs['threshold'] = threshold
#             else:
#                 self.attrs['threshold'] = (self.attrs.get('threshold') * self.n_samples + threshold * X.shape[0]) / (self.n_samples + X.shape[0])
                
#         else:
#             out = X[:, self.index]
#             out = np.where(out > self.attrs['threshold'] , 1.0, 0.0)
                        
#         self.dW = out
        

#         return out
    
    
#     def expression(self, X: List[Expr]= None) -> Expr:
#         x = Symbol(f"x{self.index}", real= True)
#         return Piecewise((1, x > self.attrs.get('threshold')), (0, True))