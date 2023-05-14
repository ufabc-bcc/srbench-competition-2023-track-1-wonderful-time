
from ..utils import create_shared_np
from ..utils.timer import timed
import numba as nb 
import ctypes

@nb.njit(nb.int64(nb.boolean[:], nb.int64))
def find_first(vec, start):
  for i in nb.prange(start, len(vec)):
    if vec[i] == 0:
      return i
  
  for i in nb.prange(0, start):
    if vec[i] == 0:
      return i
  return -1


class WeightManager:
    def __init__(self, shape: tuple):
        self.index = 0    
        self.len = shape[0]
        self.weight = create_shared_np(shape)
        self.best_weight = create_shared_np(shape)
        self.dW = create_shared_np(shape)
        self.mean = create_shared_np(shape)
        self.var = create_shared_np(shape)
        self.tree_bias = create_shared_np(self.len)
        self.allocated = create_shared_np(self.len, dtype = ctypes.c_bool)
        self.best_tree_bias = create_shared_np(self.len, val = 0)
        self.start = 0
    
    @timed
    def __next__(self):
        idx = find_first(self.allocated, self.start)
        self.start = idx + 1
        if idx == -1:
            raise ValueError("Out of space in weight manager")
        self.allocated[idx] = 1
        return idx
    
    @timed
    def free_space(self, idx: int):
        self.allocated[idx] = 0

def initWM(shape):
    global WM
    WM = WeightManager(shape)
