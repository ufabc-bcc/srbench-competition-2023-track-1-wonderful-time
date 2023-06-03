import multiprocessing as mp
from typing import List, Tuple
from ..evolution.task import SubTask
from ..utils.timer import timed
from ..utils import create_shared_np
from termcolor import colored
import traceback
import time 
from queue import Full, Empty
from ctypes import c_float
import numpy as np
from table_logger import TableLogger
import threading
from faster_fifo import Queue


SLEEP_TIME = 0.01

def execute_one_job(task, ind):    
    s = time.time()
    result = task.task.trainer.fit(ind, steps= task.task.steps_per_gen, data = task.data)
    result['time'] = time.time() - s
    return result
    
def custom_error_callback(error):
    print(''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__)))
    raise ValueError(f'Got an error from Multiprocessor: {error}')

def _put(jobs, inqueue):
    
    try:
        inqueue.put_many(jobs)
    except Full:
        pass
        
class Worker:
    def __init__(self, pid: int, metrics: dict, logger: np.ndarray, inqueue: Queue, outqueue: Queue):
        
        self.process = mp.Process(target= run_bg, args=(inqueue, outqueue, pid, metrics, logger))
        self.pid = pid
        self.process.start()
        
        
    def terminate(self):
        self.process.terminate()
        

            

class Multiprocessor:
    def __init__(self, num_workers:int = 1, chunksize: int = 10):
        self.num_workers= num_workers
        self.chunksize= chunksize
        self.train_steps = mp.Value('L', 0)
        self.nb_inqueue= mp.Value('L', 0)
        self.times = mp.Value('d', 0)
        self.processed = mp.Value('L', 0)
        
        self.inqueue = Queue()
        self.outqueue = Queue()
        self.num_workers = num_workers
        
        self.worker_logger = create_shared_np((num_workers, 9), val = 0, dtype= c_float)
        self.create_pool(num_workers= num_workers)
        
    def create_pool(self, num_workers):
        self.pool: List[Worker] = [Worker(i, {
            'train_steps': self.train_steps,
            'nb_inqueue': self.nb_inqueue,
            'times': self.times,
            'processed': self.processed,
            }, self.worker_logger, self.inqueue, self.outqueue) for i in range(num_workers)]
        
    @timed
    def log(self):
        with open('logs', 'wb') as f:
            table = TableLogger(file = f, columns = ['worker_id', 'total epochs', 'speed (epochs / s)', 'efficient time (s)', 'efficient time (%)', 'sleep time (s)',
                                                     'sleep time (%)', 'other time (%)', 'get (s)', 'backprop (s)', 'logging (s)', 'backprop speed (epoch / s)'])
            for i in range(self.num_workers):
                table(i, f'{int(self.worker_logger[i][0]):,}', #num epochs
                      f'{self.worker_logger[i][1]:.2f}', #speed
                      f'{self.worker_logger[i][2]:.2f}', #efficient time
                      f'{(self.worker_logger[i][2] / self.worker_logger[i][3] * 100):.2f}', #performance
                      f'{(self.worker_logger[i][4]):.2f}', #sleep time
                      f'{(self.worker_logger[i][4] / self.worker_logger[i][3] * 100):.2f}', #sleep time
                      f'{(100 - (self.worker_logger[i][4] + self.worker_logger[i][2]) / self.worker_logger[i][3] * 100):.2f}', #sleep time
                      f'{self.worker_logger[i][5]:.2f}', #get from queue
                      f'{self.worker_logger[i][6]:.2f}', #do one job
                      f'{self.worker_logger[i][7]:.2f}',  #log 
                      f'{self.worker_logger[i][8]:.2f}', #real back prop speed
                )
                
    
                
    
    @property
    def terminated(self):
        return self.nb_inqueue.value == 0
    
    @timed
    def __enter__(self):
        self.create_pool(num_workers= self.num_workers)
        return self
    
    
    
    @timed
    def submit_jobs(self, jobs: List[Tuple[SubTask, List]], async_submit: bool):
        with self.nb_inqueue.get_lock():
            self.nb_inqueue.value += len(jobs) 
        
        if async_submit:
            self.async_put(jobs)
        else:
            
            _put(jobs, self.inqueue)
        

            
    def async_put(self, jobs):
        
        thread = threading.Thread(target= _put, args = (jobs, self.inqueue))
        thread.start()
            
    @timed
    def __exit__(self, *args, **kwargs):
        
        for worker in self.pool:
            worker.terminate()
        
        
#Create processes running in background waiting for jobs
def run_bg(inqueue: Queue, outqueue: Queue, pid:int, metrics: dict, logger: np.ndarray):
    s = time.time()
    while True:
        try:
            start = time.time()
            job = inqueue.get()
            get = time.time()
        except Empty:
            logger[pid][4] += SLEEP_TIME
            time.sleep(SLEEP_TIME)
            print('bed time')
            
        except Exception as e:
            traceback.print_exc()
            print(colored(f'[Worker {pid}]: Error: {e}', 'red'))
            
        else:
            result = execute_one_job(*job)
            
            is_put = False
            while not is_put:
                try:
                    outqueue.put([
                        result['best_metric'],
                        result['loss'],
                        result['profile'],
                        job
                    ])
                except Full:
                    ...
                else:
                    is_put = True
            finish = time.time()
            
            
            
            #update state to display
            with metrics['nb_inqueue'].get_lock(), metrics['processed'].get_lock(), metrics['train_steps'].get_lock(), metrics['times'].get_lock():
                metrics['train_steps'].value += result['train_steps']
                metrics['nb_inqueue'].value -= 1
                metrics['processed'].value += 1
                metrics['times'].value += result['time']
            
            log = time.time()
            
            logger[pid][0] += result['train_steps']
            t = time.time()
            logger[pid][1] = logger[pid][0] / (t - s)
            logger[pid][2] += result['time']
            logger[pid][3] = t - s 
            
            logger[pid][5] = get - start
            logger[pid][6] = finish - get
            logger[pid][7] = log - finish
            logger[pid][8] = result['train_steps'] / logger[pid][6]
             