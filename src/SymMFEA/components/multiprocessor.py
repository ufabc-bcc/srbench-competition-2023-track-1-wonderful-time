import multiprocessing as mp
from typing import List, Tuple
from ..evolution.task import SubTask
from ..utils.timer import timed

def execute_one_job(args: Tuple[SubTask, List]):
    task, ind= args
    return task.task.trainer.fit(ind, task.train_dataloader, steps= task.task.steps_per_gen, val_data = task.data)

class Multiprocessor:
    def __init__(self, num_workers:int = 1, chunksize: int = 50):
        self.num_workers= num_workers
        self.chunksize= chunksize
    
    @timed
    def __enter__(self):
        self.pool = mp.Pool(self.num_workers)
        return self
    
    @timed
    def execute(self, jobs: List[Tuple[SubTask, List]]):
        metric = []
        loss = []
        train_steps = []
        for rs in self.pool.map(execute_one_job, jobs, chunksize= len(jobs) // self.num_workers):
            metric.append(rs[0])
            loss.append(rs[1])
            train_steps.append(rs[2])
        return metric, loss, train_steps    
    
    def __exit__(self, *args, **kwargs):
        self.pool.close()
        self.pool.join()
        del self.pool