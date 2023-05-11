
from ..utils.timer import timed
from ..utils import create_shared_np
import ctypes
import numpy as np
POOL_SIZE = 50000
class OffspringsPool:
    def __init__(self):
        self.pool= np.empty(POOL_SIZE, dtype = object)
        self.ready_to_collect = create_shared_np(POOL_SIZE, val=0, dtype= ctypes.c_bool)
            
    @timed
    def collect(self):
        idx = self.ready_to_collect
        rs = self.pool[idx]
        self.ready_to_collect[idx] = 0
        return rs
    
    @timed
    def append(self, inds: list):
        idx = [ind.position % POOL_SIZE for ind in inds]
        self.pool[idx] = inds
        self.ready_to_collect[idx] = 1 
        
        
class NewBorn(OffspringsPool):
    def collect_optimize_jobs(self):
        inds = self.collect()
        return [(ind.task, ind) for ind in inds]
        
    
class Optimized(OffspringsPool):
    def __init__(self, compact= False):
        super().__init__()
        self.compact= compact

    @timed
    def handle_result(self, result, optimize_jobs):
        inds = []
        metrics, loss, train_steps, profiles, attrs_list = [
            [rs[i] for rs in result] for i in range(len(result[0])) 
        ]
        for metric, job, profile, attrs in zip(metrics, optimize_jobs, profiles,attrs_list):
            task, ind = job
            if not task.is_larger_better:
                metric = -metric
            ind.set_objective(metric, compact = self.compact)
            ind.is_optimized = True
            ind.optimizer_profile = profile
            ind.setattrs(attrs)
            inds.append(ind)
            
        return inds, sum(train_steps)
    
    def create_append_callback(self, optimize_jobs, multiprocessor):
        def wrapper(result):
            inds, train_steps = self.handle_result(result, optimize_jobs= optimize_jobs)
            self.append(inds)
            multiprocessor.train_steps.value = multiprocessor.train_steps.value + train_steps
            multiprocessor.in_queue.value = multiprocessor.in_queue.value - len(inds)
            multiprocessor.processed.value = multiprocessor.processed.value + len(inds)
        
        return wrapper
    
    def collect_optimized(self, num_subpops):
        opt = self.collect()
        num = len(opt)
        if num:
            offsprings = [[] for _ in range(num_subpops)]
            
            
            for ind in opt:
                #update stats
                ind.flush_stats()
                ind.update_stats()
                ind.run_check_stats()
                ind.age = 0
                offsprings[ind.skill_factor].append(ind)
        else:
            offsprings = []
        return offsprings, num
        

def initOffspringsPool(compact= False):
    global new_born, optimized
    new_born = NewBorn()
    optimized = Optimized(compact= compact)