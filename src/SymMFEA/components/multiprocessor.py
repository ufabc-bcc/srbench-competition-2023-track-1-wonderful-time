import multiprocessing as mp
from typing import List, Tuple
from ..evolution.task import SubTask, Task
from ..utils.timer import timed
from ..utils import create_shared_np, _put, Worker, QUEUE_SIZE
from termcolor import colored
import traceback
import time 
from queue import Full, Empty
from ctypes import c_float, c_bool
import numpy as np
from prettytable import PrettyTable
import threading
from faster_fifo import Queue
from ..components.trainer import Trainer, Loss, GradOpimizer
from ..components.metrics import Metric
from typing import Dict


SLEEP_TIME = 0.01

def execute_one_job(ind, trainer, task: Task, steps_per_gen: int):    
        
    s = time.time()
    result = trainer.fit(ind, steps= steps_per_gen, data = task.task_table[ind.skill_factor].data)
    result['time'] = time.time() - s
    if not task.is_larger_better:
        result['best_metric'] = - result['best_metric']
    return result
    
def custom_error_callback(error):
    print(''.join(traceback.format_exception(etype=type(error), value=error, tb=error.__traceback__)))
    raise ValueError(f'Got an error from Multiprocessor: {error}')



class Multiprocessor:
    def __init__(self, loss: Loss, metric: Metric, task: Task,
                 optimizer: GradOpimizer, steps_per_gen: int, num_workers:int = 1, trainer_config:dict = {}):
        self.num_workers= num_workers
        self.train_steps = mp.Value('L', 0)
        self.nb_inqueue= mp.Value('L', 0)
        self.times = mp.Value('d', 0)
        self.processed = mp.Value('L', 0)
        self.stop = mp.RawValue(c_bool, False)
        self.manager = mp.Manager()
            
        self.task = task
        self.steps_per_gen:int = steps_per_gen
        self.trainer = Trainer(loss= loss, optimizer=optimizer, metric= metric, **trainer_config)
        self.inqueue = Queue(QUEUE_SIZE)
        self.outqueue = Queue(QUEUE_SIZE)
        self.num_workers = num_workers
        
        self.worker_logger = create_shared_np((num_workers, 9), val = 0, dtype= c_float)
    
                     
    def create_pool(self):
        self.pool: List[Worker] = [Worker(run_bg, self.inqueue, self.outqueue, i, {
            'train_steps': self.train_steps,
            'nb_inqueue': self.nb_inqueue,
            'times': self.times,
            'processed': self.processed,
            }, self.worker_logger, self.stop,
            steps_per_gen = self.steps_per_gen,                                          
            trainer= self.trainer, task = self.task) for i in range(self.num_workers)]
        

    @timed        
    def log(self):
        table = PrettyTable(['worker_id', 'total epochs', 'speed (epochs / s)', 'efficient time (s)', 'efficient time (%)', 'sleep time (s)',
                                                     'sleep time (%)', 'other time (%)', 'get (s)', 'backprop (s)', 'logging (s)', 'backprop speed (epoch / s)'])
        for i in range(self.num_workers):
            table.add_row([i, f'{int(self.worker_logger[i][0]):,}', #num epochs
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
            ])
        with open('logs', 'w') as f:
            f.write(str(table))
            
                
    
                
    
    @property
    def terminated(self):
        return self.nb_inqueue.value == 0
    
    @timed
    def __enter__(self):
        self.create_pool()
        return self
    
    
    
    @timed
    def submit_jobs(self, jobs: List[Tuple[SubTask, List]], async_submit: bool):
        with self.nb_inqueue.get_lock():
            self.nb_inqueue.value += len(jobs) 
        
        if async_submit:
            self.async_put(jobs)
        else:
            
            _put(jobs, inqueue= self.inqueue)
        

            
    def async_put(self, jobs):
        
        thread = threading.Thread(target= _put, args = (jobs, self.inqueue))
        thread.start()
            
    @timed
    def __exit__(self, *args, **kwargs):
        
        self.stop.value = True
        
        print("Multiprocessor's stopping...")
        del self.inqueue
        del self.outqueue
        
        for worker in self.pool:
            worker.kill()
            
        print("Multiprocessor's stopped...")
        
        
#Create processes running in background waiting for jobs
def run_bg(inqueue: Queue, outqueue: Queue, pid:int, metrics: dict, logger: np.ndarray, stop_signal: mp.RawValue, trainer: Trainer, task: Task, steps_per_gen: int):
    s = time.time()
    while not stop_signal.value:
        try:
            start = time.time()
            jobs = inqueue.get_many(max_messages_to_get = 10)
            get = time.time()
        except Empty:
            logger[pid][4] += SLEEP_TIME
            time.sleep(SLEEP_TIME)
            
        except Exception as e:
            traceback.print_exc()
            print(colored(f'[Worker {pid}]: Error: {e}', 'red'))
            
        else:
            results = []
            for job in jobs:
                result = execute_one_job(job, trainer, task, steps_per_gen)
                finish = time.time() 
                results.append([
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
            
            is_put = False
            while not is_put:
                try:
                    outqueue.put_many(results)
                    
                except Full:
                    ...
                else:
                    is_put = True