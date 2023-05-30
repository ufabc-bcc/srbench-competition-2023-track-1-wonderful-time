
from ..utils.timer import timed
from ..utils import create_shared_np
import ctypes
import numpy as np
import threading
from multiprocessing import Queue
import time
from typing import Callable
from queue import Empty
from collections.abc import Iterable 

POOL_SIZE = 50000

def run_bg(inqueue: Queue, handle:Callable):
    while True:
        try:
            rs = inqueue.get_nowait()
            
        except Empty:
            time.sleep(0.1)
            
        else:
            handle(*rs)
    

class Worker:
    def __init__(self, inqueue:Queue, handle: Callable) -> None:
        self.thread = threading.Thread(target= run_bg, args = (inqueue, handle))
        self.thread.start()
        

class OffspringsPool:
    def __init__(self):
        self.pool= np.empty(POOL_SIZE, dtype = object)
        self.ready_to_collect = create_shared_np(POOL_SIZE, val=0, dtype= ctypes.c_bool)
        self.size = 0    
    
            
    def collect(self):
        idx = self.ready_to_collect
        rs = self.pool[idx]
        self.ready_to_collect[idx] = 0
        # self.size -= len(rs)
        return rs
    
    def append(self, ind):
        if isinstance(ind, Iterable):
            idx = [i.position % POOL_SIZE for i in ind]
            self.size = self.size + len(idx)
            
        else:
            idx = ind.position % POOL_SIZE 
            self.size = self.size + 1
            
        self.pool[idx] = ind
        self.ready_to_collect[idx] = 1 
    
    def handle_input_source(self, *args):
        ...
    
    def connect_input_source(self, source: Queue):
        self.worker = Worker(source, handle = self.handle_input_source)
        
    def block_until_size_eq(self, size):
        '''
        block until self.size == size
        '''
        while self.size != size:
            time.sleep(0.1)
        
        
class NewBorn(OffspringsPool):
    def collect_optimize_jobs(self):
        inds = self.collect()
        return [(ind.task, ind) for ind in inds]
    
       
    
class Optimized(OffspringsPool):
    def __init__(self, compact= False):
        super().__init__()
        self.compact= compact
        
    
    def handle_input_source(self, metric, loss, profile, job):
        task, ind = job
        if not task.is_larger_better:
            metric = -metric
        ind.set_objective(metric, compact = self.compact)
        ind.is_optimized = True
        ind.optimizer_profile = profile
        self.append(ind)
                

    def collect_optimized(self, num_subpops):
        opt = self.collect()
        num = len(opt)
        if num:
            offsprings = [[] for _ in range(num_subpops)]
            
            
            for ind in opt:
                ind.age = 0
                offsprings[ind.skill_factor].append(ind)
        else:
            offsprings = []
        return offsprings, num
    
    
        

def initOffspringsPool(compact= False):
    global new_born, optimized
    new_born = NewBorn()
    optimized = Optimized(compact= compact)