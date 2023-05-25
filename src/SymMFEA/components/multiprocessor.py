import multiprocessing as mp
from typing import List, Tuple, Callable
from ..evolution.task import SubTask
from ..utils.timer import timed
import os
import traceback
import time 

def execute_one_job(args: Tuple[SubTask, List]):    
    s = time.time()
    task, ind= args
    result = task.task.trainer.fit(ind, steps= task.task.steps_per_gen, data = task.data)
    one_job_time = time.time() - s
    return (*result, one_job_time)


def custom_error_callback(error):
    print(''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__)))
    raise ValueError(f'Got an error from Multiprocessor: {error}')

class Multiprocessor:
    def __init__(self, num_workers:int = 1, chunksize: int = 10):
        self.num_workers= num_workers
        self.chunksize= chunksize
        self.train_steps = mp.Value('L', 0)
        self.in_queue= mp.Value('L', 0)
        self.times = mp.Value('d', 0)
        self.processed = mp.Value('L', 0)
    
    @property
    def terminated(self):
        return self.in_queue.value == 0
    
    @timed
    def __enter__(self):
        if not os.environ.get('ONE_THREAD'):
            self.pool = mp.Pool(self.num_workers)
        return self
    
    @timed
    def execute(self, jobs: List[Tuple[SubTask, List]], callback: Callable, wait_for_result: bool = False):
        with self.in_queue.get_lock():
            self.in_queue.value += len(jobs) 
        
        #remove concurrent to debug
        if os.environ.get('DEBUG') and os.environ.get('ONE_THREAD'):
            result = [execute_one_job(j) for j in jobs]
            callback(result)
        else:
            result = self.pool.map_async(execute_one_job, jobs, self.chunksize, callback=callback, error_callback= custom_error_callback)
            if wait_for_result:
                result.wait()
       
    @timed
    def __exit__(self, *args, **kwargs):
        if not os.environ.get('ONE_THREAD'):
            self.pool.close()
            self.pool.join()
