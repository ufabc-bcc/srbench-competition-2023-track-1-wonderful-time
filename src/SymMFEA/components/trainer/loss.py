
import numpy as np
import numba as nb

loss_jit = nb.njit([nb.types.Tuple((nb.float64[:], nb.float64))(nb.float64[:], nb.float64[:]),
                    nb.types.Tuple((nb.float64[:], nb.float64))(nb.float64[:], nb.float64),
                    ])

class Loss:
    def __init__(self):
        pass
    
    def __str__(self) -> str:
        pass
    
    def __call__(self, y: np.ndarray, y_hat: np.ndarray): 
        pass

@loss_jit
def mse(y: np.ndarray, y_hat: np.ndarray):
    diff = y - y_hat 
    return 2 * diff, np.mean(diff * diff )


class MSE(Loss):
    def __init__(self):
        pass
    
    
    def __str__(self) -> str:
        return 'MSE'
    
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return mse(y, y_hat)