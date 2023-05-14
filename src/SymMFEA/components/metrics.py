
import numpy as np
import numba as nb
from sklearn.metrics import r2_score
from ..utils.functional import sigmoid

metric_jit = nb.njit([nb.float64(nb.float64[:], nb.float64[:]),
                      ], cache= True)

def mse(y: np.ndarray, y_hat: np.ndarray):
    diff =  y - y_hat
    return np.mean(diff * diff)

def mae(y: np.ndarray, y_hat: np.ndarray):
    diff =  np.abs(y - y_hat)
    return np.mean(diff)

def mape(y: np.ndarray, y_hat: np.ndarray):    
    diff =  np.abs(y - y_hat / (y + 1e-12)) 
    return np.mean(diff)

def logloss(y: np.ndarray, y_hat: np.ndarray):
    y_hat = sigmoid(y_hat)
    diff = y * np.log(y_hat) + (1-y) * np.log(1 - y_hat)
    return np.mean(diff)


class Metric:
    is_larger_better: bool
    def __init__(self, func, use_numba = True, better_tol: float= 1e-3):
        self.func = metric_jit(func) if use_numba else func
        self.better_tol = better_tol
        
    def __str__(self):
        ...
    
    def is_better(self, m1:float, m2: float):
        '''
        return if m1 better than m2
        '''
        if m2 is None:
            return True
        
        return ~(self.is_larger_better ^ (m1 > m2)) & (abs(m1 - m2) > self.better_tol)
    
    def __call__(self, y: np.ndarray, y_hat: np.ndarray)-> float: 
        return self.func(y, y_hat)


class MSE(Metric):
    is_larger_better = False
    
    def __str__(self):
        return 'MSE'
    
    
    def __init__(self):
        super().__init__(mse)


class MAE(Metric):
    is_larger_better = False
    
    def __str__(self):
        return 'MAE'
    
    def __init__(self):
        super().__init__(mae)

class MAPE(Metric):
    is_larger_better = False
    
    def __str__(self):
        return 'MAPE'
    
    def __init__(self):
        super().__init__(mape)

class R2(Metric):
    is_larger_better = True
    
    def __str__(self):
        return 'R2'
    
    def __init__(self):
        super().__init__(r2_score, use_numba= False)
        
class LogLoss(Metric):
    is_larger_better = True

    def __str__(self):
        return 'LogLoss'
    
    def __init__(self):
        super().__init__(logloss, use_numba= True)