
from ..utils import create_shared_np
from ..utils.timer import timed
import numpy as np
import numba as nb 


@nb.njit(nb.int64(nb.boolean[:]))
def find_first(vec):
  for i in nb.prange(len(vec)):
    if vec[i] == 0:
      return i
  return -1


class WeightManager:
    def __init__(self, shape: tuple= (100000, 50)):
        self.index = 0    
        self.len = shape[0]
        self.weight = create_shared_np(shape)
        self.best_weight = create_shared_np(shape)
        self.dW = create_shared_np(shape)
        self.mean = create_shared_np(shape)
        self.var = create_shared_np(shape)
        self.allocated = np.zeros(shape[0], dtype = bool)
    
    @timed
    def __next__(self):
        idx = find_first(self.allocated)
        if idx == -1:
            raise ValueError("Out of space in weight manager")
        self.allocated[idx] = 1
        return idx
    
    def free_space(self, ls_idx):
        self.allocated[ls_idx] = 0

def initWM(shape):
    global WM
    WM = WeightManager(shape)
