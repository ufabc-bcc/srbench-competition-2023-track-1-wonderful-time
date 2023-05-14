
import numpy as np
import numba as nb
from ...utils.functional import sigmoid

loss_jit = nb.njit([nb.types.Tuple((nb.float64[:], nb.float64))(nb.float64[:], nb.float64[:]),
                    ])

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
    return 2 * diff, np.mean(diff * diff )


@loss_jit
def logloss(y: np.ndarray, y_hat: np.ndarray):
    y_hat = sigmoid(y_hat)
    diff = y * np.log(y_hat) + (1-y) * np.log(1 - y_hat)
    return y_hat * (1-y_hat) * (y / y_hat + (1 - y) / (1 - y_hat)), np.mean(diff)


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
        return mse(y, y_hat)