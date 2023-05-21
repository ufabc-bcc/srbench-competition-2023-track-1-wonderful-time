from ..tree import Tree
import numpy as np
from ...utils.functional import log_normalize, numba_update_adam
import numba as nb
import os

class GradOpimizer:
    def __init__(self, lr: float = 1e-2, weight_decay: float=0):
        self.lr = lr 
        self.weight_decay = weight_decay
    
        
    def backprop(self, tree: Tree, dY: float, **kwargs):
        stack = []
        root = tree.nodes[-1]
        
        bp = root.backprop(dY)
        
        if tree.length == 1:
            return {}
        
        stack.extend(bp)
            
        for i in range(2, len(tree.nodes) + 1):
            node = tree.nodes[-i]
            dY = stack.pop()  
            dY = node.backprop(dY)
            
            if not node.is_leaf:
                stack.extend(dY)
            
        dW = self.compute_gradient(tree)
        
        W = tree.W 
        
        W[:] = W - dW * self.lr 
        
        return {}
    
    def compute_gradient(self, tree):
        dW = log_normalize(tree.dW * (1 + self.weight_decay))
        return dW
    

#===========================================ADAM==============================================

@numba_update_adam
def update_m(m, beta, g, t):
    m = m * beta + (1 - beta) * g 
    m_hat = m / (1 - np.power(beta, t))
    return m, m_hat
    
@numba_update_adam
def update_v(v, beta, g, t):
    v = v * beta + (1 - beta) * g * g  
    v_hat = v / (1 - np.power(beta, t))
    return v, v_hat


@nb.njit(nb.float64(nb.float64, nb.float64[:], nb.float64),
    cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
def caculate_bias(b, dY, lr):
    return b - np.mean(dY) * lr

@nb.njit(nb.float64[:](nb.float64[:], nb.float64, nb.float64[:], nb.float64[:], nb.float64),
    cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
def caculate_W(W, lr, m_hat, v_hat, eps):
    return W - lr * m_hat / (np.sqrt(v_hat) + eps)


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
        
        assert isinstance(profile, dict)
        if len(profile) == 0:
            profile = {
                        'step': 1,
                        'lr': self.lr,
                       }
        
        #update bias
        tree.set_bias(caculate_bias(tree.bias, dY, profile['lr']))
        
        
        bp = root.backprop(dY)
        
        if tree.length == 1:
            return profile
        
        stack.extend(bp)
            
        for i in range(2, len(tree.nodes) + 1):
            node = tree.nodes[-i]
            dY = stack.pop()  
            dY = node.backprop(dY)
            
            if not node.is_leaf:
                stack.extend(dY)
        
        #compute gradient 
        dW = self.compute_gradient(tree)
        
        momentum = tree.momentum
        velocity = tree.velocity
        
        momentum[:], mw_hat = update_m(momentum, self.betas1, dW, profile['step'])
        velocity[:], vw_hat = update_v(velocity, self.betas2, dW, profile['step'])
        
        
        W = tree.W 
        W[:] = caculate_W(W, profile['lr'], mw_hat, vw_hat, self.eps)
        
        profile['step'] = profile['step'] + 1
        return profile