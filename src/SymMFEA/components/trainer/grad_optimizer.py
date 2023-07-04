from ..tree import Tree
import numpy as np
from ...utils.functional import log_normalize, numba_update_adam, ONE
import numba as nb
import os

class GradOpimizer:
    def __init__(self, lr: float = 1e-2, weight_decay: float=0):
        self.lr = np.float32(lr)
        self.weight_decay = np.float32(weight_decay)
    
        
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
        dW = tree.dW * (ONE + self.weight_decay)
        return dW
    

#===========================================ADAM==============================================

@numba_update_adam
def update_m(m, beta, g, t):
    m = m * beta + (ONE - beta) * g 
    m_hat = m / (ONE - np.power(beta, t))
    return m, m_hat
    
@numba_update_adam
def update_v(v, beta, g, t):
    v = v * beta + (ONE - beta) * g * g  
    v_hat = v / (ONE - np.power(beta, t))
    return v, v_hat


@nb.njit(nb.float32(nb.float32, nb.float32[:], nb.float32),
    cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
def caculate_bias(b, dY, lr):
    return b - np.mean(dY) * lr

@nb.njit(nb.float32[:](nb.float32[:], nb.float32, nb.float32[:], nb.float32[:], nb.float32),
    cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
def caculate_W(W, lr, m_hat, v_hat, eps):
    return W - lr * m_hat / (np.sqrt(v_hat) + eps)


class ADAM(GradOpimizer):
    def __init__(self, lr: float = 1e-2, betas=(0.9, 0.999), eps: float =1e-08, weight_decay: float = 0):
        super(ADAM, self).__init__(lr= lr, weight_decay=weight_decay)
        self.betas1:float = np.float32(betas[0])
        self.betas2:float = np.float32(betas[1])
        self.eps = eps 

    def backprop(self, tree: Tree, dY: float, profile: dict, **kwargs):
        # stack = []
        if tree.length <= 1:
            return profile
        
        root = tree.nodes[-1]
        stack = np.empty((tree.length, root.dX.shape[1]), dtype = np.float32)
        
        
        
        
        assert isinstance(profile, dict)
        if len(profile) == 0:
            profile = {
                        'step': 1,
                        'lr': self.lr,
                       }
        
        #update bias
        tree.set_bias(caculate_bias(tree.bias, dY, profile['lr']))
        stack[:root.arity] = root.backprop(dY)
        top = root.arity - 1
                
        if tree.length == 1:
            return profile
        
            
        for i in range(2, len(tree.nodes) + 1):
            node = tree.nodes[-i]
            dY = stack[top]
            top -= 1
            dY = node.backprop(dY)
              
            if not node.is_leaf:
                stack[top + 1 : top + 1 + node.arity] = dY
                top += node.arity
        
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