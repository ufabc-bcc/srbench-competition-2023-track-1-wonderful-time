
from ..utils import create_shared_np


class WeightManager:
    def __init__(self, shape: tuple= (1000000, 50)):
        self.index = 0    
        self.len = shape[0]
        self.weight = create_shared_np(shape)
        self.best_weight = create_shared_np(shape)
        self.dW = create_shared_np(shape)
    
    def __next__(self):
        tmp = self.index
        self.index += 1
        return tmp

def initWM(shape):
    global WM
    WM = WeightManager(shape)
