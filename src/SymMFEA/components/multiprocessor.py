import multiprocessing as mp
from typing import List, Tuple, Callable
from ..evolution.task import SubTask
from ..utils.timer import timed

def execute_one_job(args: Tuple[SubTask, List]):
    task, ind= args
    return task.task.trainer.fit(ind, task.train_dataloader, steps= task.task.steps_per_gen, val_data = task.data)

def custom_error_callback(error):
    raise ValueError(f'Got an error: {error}')

class Multiprocessor:
    def __init__(self, num_workers:int = 1, chunksize: int = 10):
        self.num_workers= num_workers
        self.chunksize= chunksize
        self.train_steps = mp.Value('L', 0)
        self.in_queue= mp.Value('L', 0)
        self.processed = mp.Value('L', 0)
    
    @property
    def terminated(self):
        return self.in_queue.value == 0
    
    @timed
    def __enter__(self):
        self.pool = mp.Pool(self.num_workers)
        return self
    
    @timed
    def execute(self, jobs: List[Tuple[SubTask, List]], callback: Callable, wait_for_result: bool = False):
        self.in_queue.value = self.in_queue.value + len(jobs) 
        result = self.pool.map_async(execute_one_job, jobs, self.chunksize, callback=callback, error_callback= custom_error_callback)
        if wait_for_result:
            result.wait()
    
    def __exit__(self, *args, **kwargs):
        self.pool.close()
        self.pool.join()
