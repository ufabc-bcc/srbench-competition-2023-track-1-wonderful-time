
from ..utils import create_shared_np
import ctypes
import numpy as np
import threading
from faster_fifo import Queue
import time
from typing import Callable
from queue import Empty
from collections.abc import Iterable 

POOL_SIZE = 50000

def run_bg(inqueue: Queue, handle:Callable, stop_signal: threading.Event):
    while not stop_signal.is_set():
        try:
            results = inqueue.get_many()
            
        except Empty:
            time.sleep(0.1)
            
        else:
            for rs in results:
                handle(*rs)
    

class Worker:
    def __init__(self, inqueue:Queue, handle: Callable) -> None:
        self.stop = threading.Event()
        self.thread = threading.Thread(target= run_bg, args = (inqueue, handle, self.stop))
        self.thread.start()
        
        
    def kill(self):
        self.stop.set()
        self.thread.join()
        
class PseudoLock:
    def __enter__(*args, **kwargs):
        ...
    
    def __exit__(*args, **kwargs):
        ... 
class OffspringsPool:
    def __init__(self, race_safe= False):
        self.pool= np.empty(POOL_SIZE, dtype = object)
        self.ready_to_collect = create_shared_np(POOL_SIZE, val=0, dtype= ctypes.c_bool, race_safe= False)
        self.size = 0    
        self.race_safe = race_safe
        
        if self.race_safe:
            self.lock = threading.Lock()
        else:
            self.lock = PseudoLock()
            
    def collect(self):
        with self.lock:
            idx = self.ready_to_collect
            rs = self.pool[idx]
            self.ready_to_collect[idx] = 0
            self.size -= len(rs)
            return rs
    
    def append(self, ind):
        with self.lock:
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
        
    def kill(self):
        if hasattr(self, 'worker'):
            self.worker.kill()
        
    def block_until_size_eq(self, size):
        '''
        block until self.size == size
        '''
        while self.size != size:
            time.sleep(0.1)
        
        
class NewBorn(OffspringsPool):
    def collect_optimize_jobs(self):
        inds = self.collect()
        return inds.tolist()
    
       
    
class Optimized(OffspringsPool):
    def __init__(self, compact= False):
        super().__init__(race_safe= True)
        self.compact= compact
        
    
    def handle_input_source(self, metric, loss, profile, ind):
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
    
def terminateOffspringsPool():
    global new_born, optimized
    new_born.kill()
    optimized.kill()