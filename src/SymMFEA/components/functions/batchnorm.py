from typing import List
from sympy import Expr
from .node import Node	
import numpy as np	
import numba as nb	

@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(nb.float64[:, :], nb.float64, nb.float64))	
def batchnorm(x, mean, var):	
    x = np.ravel(x)	
    scale = 1 / (np.sqrt(var) + 1e-12)
    return (x - mean) * scale, np.full((1, x.shape[0]), scale.item())	

@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(nb.float64[:, :]))	
def batchnorm_training(x):	
    x = np.ravel(x)	
    mean = np.mean(x)
    std = np.std(x)
    scale = 1 / (std + 1e-12)
    return (x - mean) * scale, np.full((1, x.shape[0]), scale.item())	



class BatchNorm(Node):	
    is_nonlinear = True	
    def __init__(self, **kwargs):	
        super().__init__(arity = 1)	

    def __str__(self) -> str:	
        return 'BN'	

    def __call__(self, X, update_stats= False):	
        if abs(self.mean) <1e-15 and abs(self.var) < 1e-15:
            out, self.dX = batchnorm_training(X)
        else:
            out, self.dX = batchnorm(X, self.mean, self.var)	

        self.dW = out	

        assert self.dX.ndim == 2, self.dX.ndim	
        if update_stats:	
            self.update_stats(out)	
        return out
    
    def expression(self, X: List[Expr]) -> Expr:
        return (X[0] - self.mean) / (np.sqrt(self.var) + 1e-12)