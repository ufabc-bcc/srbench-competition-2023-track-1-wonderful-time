
import numpy as np
import numba as nb
from ...utils.functional import sigmoid
import os 
EPS = np.finfo(np.float64).eps
loss_jit = nb.njit([nb.types.Tuple((nb.float64[:], nb.float64))(nb.float64[:], nb.float64[:]),
                    ], cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)

class Loss:
    def __init__(self):
        ...
    
    def __str__(self) -> str:
        ...
    
    def __call__(self, y: np.ndarray, y_hat: np.ndarray): 
        ...

@loss_jit
def mse(y: np.ndarray, y_hat: np.ndarray):
    diff = y_hat - y 
    return 2 * diff, np.mean(diff * diff)



@loss_jit
def logloss(y: np.ndarray, y_hat: np.ndarray):
    sig = sigmoid(y_hat)
    diff = -(y * np.log(sig) + (1-y) * np.log(1 - sig))
    # -sig * (1-sig) * (y / (sig) + (1 - y) / (1 - sig)) 
    return  y * (sig - 1) - (y - 1) * sig,  np.mean(diff)


class MSE(Loss):
    def __init__(self):
        ...
    
    
    def __str__(self) -> str:
        return 'MSE'
    
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return mse(y, y_hat)
    
class LogLossWithSigmoid(Loss):
    def __init__(self):
        ...
    
    
    def __str__(self) -> str:
        return 'LogLoss'
    
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        assert np.amax(y) <=1 and np.amin(y) >=0
        return logloss(y, y_hat)