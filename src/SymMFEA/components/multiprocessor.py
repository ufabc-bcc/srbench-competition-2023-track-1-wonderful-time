import multiprocessing as mp
from typing import List, Tuple, Callable
from ..evolution.task import SubTask
from ..utils.timer import timed
from termcolor import colored
import traceback
import time 
from queue import Full, Empty

def execute_one_job(task, ind):    
    s = time.time()
    result = task.task.trainer.fit(ind, steps= task.task.steps_per_gen, data = task.data)
    result['time'] = time.time() - s
    return result
    
def custom_error_callback(error):
    print(''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__)))
    raise ValueError(f'Got an error from Multiprocessor: {error}')



        
class Worker:
    def __init__(self, inqueue: mp.JoinableQueue, outqueue: mp.Queue, pid: int, metrics: dict):
        
        self.process = mp.Process(target= run_bg, args=(inqueue, outqueue, pid, metrics))
        
        self.process.start()
        
    def kill(self):
        self.process.kill()
            

class Multiprocessor:
    def __init__(self, num_workers:int = 1, chunksize: int = 10):
        self.num_workers= num_workers
        self.chunksize= chunksize
        self.train_steps = mp.Value('L', 0)
        self.nb_inqueue= mp.Value('L', 0)
        self.times = mp.Value('d', 0)
        self.processed = mp.Value('L', 0)
        
        self.inqueue = mp.JoinableQueue()
        self.outqueue = mp.Queue()
        self.create_pool(num_workers= num_workers)
        
    def create_pool(self, num_workers):
        self.pool: List[Worker] = [Worker(self.inqueue, self.outqueue, i, {
            'train_steps': self.train_steps,
            'nb_inqueue': self.nb_inqueue,
            'times': self.times,
            'processed': self.processed,
            }) for i in range(num_workers)]
    
    @property
    def terminated(self):
        return self.nb_inqueue.value == 0
    
    @timed
    def __enter__(self):
        self.create_pool(num_workers= self.num_workers)
        return self
    
    @timed
    def submit_jobs(self, jobs: List[Tuple[SubTask, List]]):
        with self.nb_inqueue.get_lock():
            self.nb_inqueue.value += len(jobs) 
        

        for job in jobs:
            try:
                self.inqueue.put_nowait(job)
            except Full:
                pass
                
                
    @timed
    def __exit__(self, *args, **kwargs):
        self.inqueue.close()
        self.outqueue.close()
        
        for worker in self.pool:
            worker.kill()
        
        
#Create processes running in background waiting for jobs
def run_bg(inqueue: mp.JoinableQueue, outqueue: mp.Queue, pid:int, metrics: dict):
    while True:
        try:
            job = inqueue.get_nowait()
        except Empty:
            time.sleep(0.001)
            
        except Exception as e:
            traceback.print_exc()
            print(colored(f'[Worker {pid}]: Error: {e}', 'red'))
            
        else:
            result = execute_one_job(*job)
            inqueue.task_done()
            outqueue.put([
                result['best_metric'],
                result['loss'],
                result['profile'],
                job
            ])
            
            #update state to display
            with metrics['nb_inqueue'].get_lock(), metrics['processed'].get_lock(), metrics['train_steps'].get_lock(), metrics['times'].get_lock():
                metrics['train_steps'].value += result['train_steps']
                metrics['nb_inqueue'].value -= 1
                metrics['processed'].value += 1
                metrics['times'].value += result['time']
            