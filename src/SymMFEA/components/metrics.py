
import numpy as np
import numba as nb
from sklearn.metrics import r2_score

metric_jit = nb.njit([nb.float64(nb.float64[:], nb.float64[:]),
                      nb.float64(nb.float64[:], nb.float64),
                      ])

def mse(y: np.ndarray, y_hat: np.ndarray):
    diff =  y - y_hat
    return np.mean(diff * diff)

def mae(y: np.ndarray, y_hat: np.ndarray):
    diff =  np.abs(y - y_hat)
    return np.mean(diff)

def mape(y: np.ndarray, y_hat: np.ndarray):    
    diff =  np.abs(y - y_hat / (y + 1e-12)) 
    return np.mean(diff)
class Metric:
    is_larger_better: bool
    def __init__(self, func, is_numba = True):
        self.func = metric_jit(func) if is_numba else func
        
    def __str__(self):
        ...
    
    @classmethod
    def is_better(cls, m1, m2):
        '''
        return if m1 better than m2
        '''
        return ~(cls.is_larger_better ^ (m1 > m2))
    
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
        super().__init__(r2_score, is_numba= False)