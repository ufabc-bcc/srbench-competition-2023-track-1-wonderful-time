
from typing import List
from ..utils.timer import timed
from ..utils import create_shared_np
import multiprocessing as mp
import ctypes
import numpy as np
MAX_SIZE = 1000000
class OffspringsPool:
    def __init__(self):
        self.pool= np.empty(MAX_SIZE, dtype = object)
        self.ready_to_collect = create_shared_np(MAX_SIZE, val=0, dtype= ctypes.c_bool)
            
    @timed
    def collect(self):
        idx = self.ready_to_collect
        rs = self.pool[idx]
        self.ready_to_collect[idx] = 0
        return rs
    
    @timed
    def append(self, inds: list):
        idx = [ind.position for ind in inds]
        self.pool[idx] = inds
        self.ready_to_collect[idx] = 1 
        
        
class NewBorn(OffspringsPool):
    def collect_optimize_jobs(self):
        inds = self.collect()
        return [(ind.task, ind) for ind in inds]
        
    
class Optimized(OffspringsPool):
    def __init__(self):
        super().__init__()
        self.train_steps = mp.Value('L', 0)
        
    @staticmethod
    def handle_result(result, optimize_jobs):
        inds = []
        metrics, loss, train_steps = [
            [rs[i] for rs in result] for i in range(3) 
        ]
        for metric, job in zip(metrics, optimize_jobs):
            task, ind= job
            ind.objective = [metric if task.is_larger_better else -metric]
            ind.is_optimized = True
            inds.append(ind)
            
        return inds, sum(train_steps)
    
    def create_append_callback(self, optimize_jobs):
        def wrapper(result):
            inds, train_steps = self.handle_result(result, optimize_jobs= optimize_jobs)
            self.append(inds)
            self.train_steps.value = self.train_steps.value + train_steps
        
        return wrapper
    
    def collect_optimized(self, num_subpops):
        opt = self.collect()
        num = len(opt)
        if num:
            offsprings = [[] for _ in range(num_subpops)]
            
            
            for ind in opt:
                offsprings[ind.skill_factor].append(ind)
        else:
            offsprings = []
        return offsprings, num
        

def initOffspringsPool():
    global new_born, optimized
    new_born = NewBorn()
    optimized = Optimized()