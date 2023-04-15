
from typing import Dict, List
from ..utils.timer import timed
import multiprocessing as mp

class OffspringsPool:
    def __init__(self):
        global manager
        self.pool: Dict[int, object] = manager.dict()
    
    @timed
    def append_offsprings(self, inds: List[object]):
        for ind in inds:
            self.pool[ind.position] = ind
        
    
    def remove(self, idx):
        for i in idx:
            del self.pool[i]

        
class NewBorn(OffspringsPool):
    def collect_optimize_job(self):
        
        idx = self.pool.keys()
        jobs = [
            (self.pool[i].task, self.pool[i]) for i in idx
        ]
        
        self.remove(idx)
        
        return jobs
    
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
            self.append_offsprings(inds)
            self.train_steps.value = self.train_steps.value + train_steps
        
        return wrapper
    
    def collect_optimized(self, num_subpops):
        num = len(self.pool)
        if num == 0:
            return [], num
        
        offsprings = [[] for _ in range(num_subpops)]
        
        idx = self.pool.keys()
        for i in idx:
            o = self.pool[i]
            offsprings[o.skill_factor].append(o)
        
        self.remove(idx)
        return offsprings, num
        

def initOffspringsPool():
    global new_born, optimized, manager
    manager = mp.Manager()
    new_born = NewBorn()
    optimized = Optimized()