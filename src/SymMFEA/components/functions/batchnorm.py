# from typing import List
# from sympy import Expr
# from .node import Node
# import numpy as np	
# import numba as nb	

# @nb.njit(nb.types.Tuple((nb.float32[:], nb.float32[:, :]))(nb.float32[:, :]))	
# def batchnorm_training(x):	
#     x = np.ravel(x)	
#     mean = np.mean(x)
#     std = np.std(x)
#     scale = 1 / (std + 1e-12)
#     return (x - mean) * scale, np.full((1, x.shape[0]), scale.item())	

# #NOTE: batchnorm deprecated

# class BatchNorm(Node):	
#     '''
#     value is scale (1 / std) and bias is -mean / std
#     '''
#     is_nonlinear = True	
#     def __init__(self, **kwargs):	
#         super().__init__(arity = 1)	
#         #do not update value via gradient descent
#         self.dW = 0
#         self.inp_mean = 0.0 
#         self.inp_var = 0.0

#     def __str__(self) -> str:	
#         return 'BN'	
    

#     def __call__(self, X, training= False):	
#         if training:
#             out, self.dX = batchnorm_training(X)
            
#             #update stats
#             self.inp_mean, self.inp_var, _= update_node_stats(self.inp_mean, self.inp_var, self.n_samples, np.ravel(X))
            
#             self.update_value_hard(1 / (np.sqrt(self.inp_var) + 1e-12))
#             self.set_bias_hard(- self.inp_mean / np.sqrt(self.inp_var + 1e-12))
            
#             assert self.dX.ndim == 2, self.dX.ndim	
#         else:
#             out = np.ravel(X)

        	
            	
#         return out
    
#     def expression(self, X: List[Expr]) -> Expr:
#         return X[0]