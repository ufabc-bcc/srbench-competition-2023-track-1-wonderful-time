from ..tree import Tree
import numpy as np
from ...utils.functional import normalize
class GradOpimizer:
    def __init__(self, lr: float = 1e-2):
        self.lr = lr 
    
    def update_lr(self, lr):
        self.lr = lr
        
    def backprop(self, tree: Tree, dY: float):
        stack = []
        root = tree.nodes[-1]
        
        bp = root.backprop(dY, lr = self.lr)
        
        if tree.length == 1:
            return
        
        stack.extend(bp)
            
        for i in range(2, len(tree.nodes) + 1):
            node = tree.nodes[-i]
            dY = stack.pop()  
            dY = node.backprop(dY, lr = self.lr)
            
            if not node.is_leaf:
                stack.extend(dY)
            
        dW = normalize(tree.dW * tree.node_grad_mask )
        dB = normalize(tree.dB * tree.node_grad_mask )
        
        tree.W = tree.W + dW * self.lr 
        tree.bias = tree.bias + dB * self.lr


class ADAM(GradOpimizer):
    def __init__(self, lr: float = 1e-2, betas=(0.9, 0.999), eps=1e-08):
        self.lr = lr
        
    
    def update_lr(self, lr):
        self.lr = lr
        
    def backprop(self, tree: Tree, dY: float):
        stack = []
        root = tree.nodes[-1]
        
        bp = root.backprop(dY, lr = self.lr)
        
        if tree.length == 1:
            return
        
        stack.extend(bp)
            
        for i in range(2, len(tree.nodes) + 1):
            node = tree.nodes[-i]
            dY = stack.pop()  
            dY = node.backprop(dY, lr = self.lr)
            
            if not node.is_leaf:
                stack.extend(dY)
            
        dW = normalize(tree.dW * tree.node_grad_mask )
        dB = normalize(tree.dB * tree.node_grad_mask )
                
        tree.W = tree.W + dW * self.lr 
        tree.bias = tree.bias + dB * self.lr