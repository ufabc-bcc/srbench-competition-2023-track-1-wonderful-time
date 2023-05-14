from __future__ import annotations
from typing import List
import numpy as np
import numba as nb
from sympy import Expr

@nb.njit(nb.float64[:, :](nb.float64[:,:], nb.float64[:]), cache= True)
def matrix_vec_prod(m, v):
    result = np.empty_like(m, dtype = np.float64)
    for i in nb.prange(m.shape[0]):
        result[i] = m[i] * v
    return result

    
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
        
        
        self.dW:np.ndarray
        self.dX:List[np.ndarray]
        
        self.tree = None
        self.compiled: bool = False
        self.attrs: dict = dict()
        
    
    def __call__(self, X, training:bool= False):
        ...
    
    
    def expression(self, X: List[Expr]) -> Expr:
        ...
        
    @property
    def n_samples(self):
        return self.tree.n_samples
    
    
    @staticmethod
    def deepcopy(node: Node, new_class = None):
        
        new_node = node.__class__(arity= node.arity, index= node.index) if new_class is None else new_class(arity= node.arity, index= node.index)
        new_node.attrs = node.attrs
        

        if node.tree is not None:
            new_node.value = node.tree.W[node.id]
            
            
        return new_node
    
    def __str__(self) -> str:
        return 
        
    def backprop(self, dY, lr) -> List[float]:
        #receive dY from parent and pass new a list of dLs to children
        
        self.tree.dW[self.id] = np.mean(self.dW * dY) 
        
        #len self.dX = arity
        if self.is_leaf:
            return None
        return matrix_vec_prod(self.dX, dY)
         
    def update_value_hard(self, val: float):
        self.value = val
        self.tree.W[self.index] = val
    
    @property
    def mean(self):
        return self.tree.mean[self.id]
    
    @property
    def var(self):
        return self.tree.var[self.id]
        
    @property
    def is_leaf(self) -> bool:
        return self.arity == 0

    def compile(self, tree= None):
        self.tree = tree
        self.compiled = True
        