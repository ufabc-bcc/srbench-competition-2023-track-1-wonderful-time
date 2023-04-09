from __future__ import annotations
from typing import List
import numpy as np
import numba as nb

@nb.njit
def matrix_vec_prod(m, v):
    result = np.empty_like(m, dtype = np.float64)
    for i in nb.prange(m.shape[0]):
        result[i] = m[i] * v
    return result
class Node:
    is_nonlinear = False
    def __init__(self, 
                 arity: int = 1,
                #  value: float = 0,
                #  bias: float = 0,
                #  node_type: int = 1,
                #  depth: int =  1,
                #  lenth: int = 1,
                #  parent: int = 1,
                #  level: int = 1,
                 
                #  id: int = 1,
                 
                 **kwargs):
        r'''
        parent: index of parent node
        level: length of the path to the root node, not including the node itself
        value for constants or weighting factor for variables
        '''
        
        self.id:int
        self.depth:int
        self.length:int
        self.parent:int
        # self.level = level
        # self.node_type = node_type
        
        self.arity:int = arity
        
        self.value:float = np.random.normal(0, 1, 1).item()
        
        self.bias:float = np.random.normal(0, 1, 1).item()
        
        self.dW:np.ndarray
        self.dX:List[np.ndarray]
        
        self.tree = None
        self.compiled: bool = False
        
    @staticmethod
    def deepcopy(node: Node, new_class = None):
        assert node.compiled, 'make sure to compile node before copy'
        
        new_node = node.__class__() if new_class is None else new_class()
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
        
        
        
    @property
    def is_leaf(self) -> bool:
        return self.arity == 0

    def compile(self, tree):
        self.tree = tree
        self.compiled = True
        
