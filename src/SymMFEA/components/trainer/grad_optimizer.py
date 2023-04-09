from ..tree import Tree
import numpy as np
from ...utils.functional import log_normalize
from numba import jit


#NOTE: syntax outdated
class GradOpimizer:
    def __init__(self, lr: float = 1e-2, weight_decay: float=0):
        self.lr = lr 
        self.weight_decay = weight_decay
    
    def update_lr(self, lr):
        self.lr = lr
        
    def backprop(self, tree: Tree, dY: float, **kwargs):
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
            
        dW, dB = self.compute_gradient(tree)
        
        tree.W = tree.W - dW * self.lr 
        tree.bias = tree.bias - dB * self.lr
    
    def compute_gradient(self, tree):
        dW = log_normalize(tree.dW * (tree.node_grad_mask + self.weight_decay))
        dB = log_normalize(tree.dB * (tree.node_grad_mask + self.weight_decay))
        return dW, dB
    


@jit(nopython = True)
def update_m(m, beta, g, t):
    m = m * beta + (1 - beta) * g 
    m_hat = m / (1 - np.power(beta, t))
    return m, m_hat
    
@jit(nopython = True)
def update_v(v, beta, g, t):
    v = v * beta + (1 - beta) * g * g  
    v_hat = v / (1 - np.power(beta, t))
    return v, v_hat


class ADAM(GradOpimizer):
    def __init__(self, lr: float = 1e-2, betas=(0.9, 0.999), eps: float =1e-08, weight_decay: float = 0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas1:float = betas[0]
        self.betas2:float = betas[1]
        self.eps = eps 

    def backprop(self, tree: Tree, dY: float, profile: dict, **kwargs):
        stack = []
        root = tree.nodes[-1]
        
        if len(profile) == 0:
            profile = {
                        'step': 1,
                        'mw': 0,
                        'vw': 0,
                        'mb': 0,
                        'vb': 0,
                        'lr': self.lr,
                       }
        
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
        
        #compute gradient 
        dW, dB = self.compute_gradient(tree)
        
        profile['mw'], mw_hat = update_m(profile['mw'], self.betas1, dW, profile['step'])
        profile['mb'], mb_hat = update_m(profile['mb'], self.betas1, dB, profile['step'])
        
        profile['vw'], vw_hat = update_v(profile['vw'], self.betas2, dW, profile['step'])
        profile['vb'], vb_hat = update_v(profile['vb'], self.betas2, dB, profile['step'])
        
                
        tree.W = tree.W - profile['lr'] * mw_hat / (np.sqrt(vw_hat) + self.eps)
        tree.bias = tree.bias - profile['lr'] * mb_hat / (np.sqrt(vb_hat) + self.eps)
        
        profile['step'] = profile['step'] + 1