from __future__ import annotations
from typing import List
import numpy as np
import numba as nb
from sympy import Expr
from ...utils.timer import timed

@nb.njit(nb.float64[:, :](nb.float64[:,:], nb.float64[:]))
def matrix_vec_prod(m, v):
    result = np.empty_like(m, dtype = np.float64)
    for i in nb.prange(m.shape[0]):
        result[i] = m[i] * v
    return result

@timed(nb.njit(nb.types.Tuple((nb.float64, nb.float64, nb.int64))(nb.float64, nb.float64, nb.int64, nb.float64[:])))
def update_stats(old_mean, old_var, old_samples, X):	
    new_samples = X.shape[0] + old_samples	
    mean = (old_mean * old_samples + X.sum()) / new_samples	
    var = (old_var * old_samples + X.var() * X.shape[0]) / new_samples	
    return mean, var, new_samples

    
class Node:
    is_nonlinear = False
    def __init__(self, 
                 arity: int = 1,   
                 index: int = -1,              
                 **kwargs):
        self.index = index
        self.id:int
        self.depth:int
        self.length:int
        self.parent:int
        self.arity:int = arity
        
        self.value:float = 1
        
        self.bias:float = 0
        
        self.dW:np.ndarray
        self.dX:List[np.ndarray]
        
        self.tree = None
        self.compiled: bool = False
        self.mean: float= 0
        self.var: float= 0
        self.inp_mean: float= 0
        self.inp_var: float= 0
        self.n_samples:int = 0
        
    
    def __call__(self, X, update_stats: bool= False):
        ...
    
    
    def expression(self, X: List[Expr]) -> Expr:
        ...
        
    def flush_stats(self):
        self.mean = 0
        self.var = 0
        self.n_samples = 0
        
    @staticmethod
    def deepcopy(node: Node, new_class = None, copy_batchnorm= False):
        assert node.compiled, 'make sure to compile node before copy'
        
        new_node = node.__class__(arity= node.arity, index= node.index) if new_class is None else new_class(arity= node.arity, index= node.index)
        new_node.mean = node.mean
        new_node.var = node.var
        
        if copy_batchnorm:
            new_node.inp_mean = node.inp_mean
            new_node.inp_var = node.inp_var
        
        if node.tree is not None:
            new_node.value = node.tree.W[node.id]
            new_node.bias = node.tree.bias[node.id] 
            
            
        return new_node
    
    def __str__(self) -> str:
        return 
        
    def backprop(self, dY, lr) -> List[float]:
        #receive dY from parent and pass new a list of dLs to children
        
        self.tree.dW[self.id] = np.mean(self.dW * dY) 
        self.tree.dB[self.id] = np.mean(dY)
        
        #len self.dX = arity
        if self.is_leaf:
            return None
        return matrix_vec_prod(self.dX, dY)
    
    def update_stats(self, X):	
        '''	
        X: 1d array	
        '''	
        self.mean, self.var, self.n_samples = update_stats(self.mean, self.var, self.n_samples, X)
    
    def update_inp_stats(self, X):	
        '''	
        need to update_stats first
        X: 1d array	
        '''	
        self.inp_mean, self.inp_var, _ = update_stats(self.inp_mean, self.inp_var, self.n_samples, X)
        
    @property
    def is_leaf(self) -> bool:
        return self.arity == 0

    def compile(self, tree= None):
        self.tree = tree
        self.compiled = True
        