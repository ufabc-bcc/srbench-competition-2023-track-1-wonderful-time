from typing import List
from ..components.trainer import Trainer
from ..components.data_pool import DataPool, DataView, TrainDataLoader, initDataPool
from ..components.trainer import Loss, GradOpimizer
from ..components.metrics import Metric
import numpy as np
from ..utils import FinetuneProgressBar
from ..utils.timer import *
from ..components import data_pool

class Task:
    def __init__(self, X: np.ndarray, y:np.ndarray, loss: Loss,
                 optimizer: GradOpimizer, metric: Metric, steps_per_gen: int, batch_size:int,
                 shuffle: bool, test_size: float = 0.2):
        assert len(X.shape) == 2
        self.terminal_set = [i for i in range(X.shape[1])]
        
        #init datapool
        initDataPool(X, y, test_size= test_size)
        self.data_pool = data_pool.data_pool
        # self.data_pool = DataPool(X, y, test_size= test_size)
        
        self.trainer = Trainer(loss= loss, optimizer=optimizer, metric= metric)
        self.train_dataloader_cfg = {
            'batch_size': batch_size,
            'shuffle': shuffle,
        }
        self.steps_per_gen:int = steps_per_gen
        self.is_larger_better = metric.is_larger_better
        
    
class SubTask:
    def __init__(self, task: Task, data_sample:float = 1, terminal_set:List[int] = None):
        if terminal_set is None:
            self.terminal_set = task.terminal_set 
        self.task = task
        self.data = DataView(task.data_pool, data_sample)
        self.train_dataloader = TrainDataLoader(self.data, **self.task.train_dataloader_cfg)
        self.is_larger_better = self.task.is_larger_better
    
    @timed
    def train(self, ind, steps = None):            
        return self.task.trainer.fit(ind, self.train_dataloader, steps= self.task.steps_per_gen if steps is None else steps, val_data = self.data)
        
        
    def finetune(self, ind, finetune_steps: int = 5000, decay_lr: float = 100, verbose = False):
        
        return 
        self.flush_history(ind)
        if finetune_steps < 1:
            return
        
        lr= self.task.trainer.optimizer.lr 
        self.task.trainer.update_lr(self.task.trainer.optimizer.lr / decay_lr)
        
        pbar = FinetuneProgressBar(
            num_iters= finetune_steps,
            metric_name= [str(self.task.trainer.loss), str(self.task.metric)]
        ) if verbose else range(finetune_steps)
        
        for step in pbar.pbar if verbose else pbar:
            metric, loss = self.task.trainer.fit(ind, self.train_dataloader, 1)
            self.update_learning_state(ind, metric)
            
            if verbose:
                pbar.update(loss= loss, metric = metric, best_metric = ind.best_metric, reverse = not self.task.metric.is_larger_better)
            
        ind.rollback_best()
        
        # assert self.eval(ind, bypass_check= True) == ind.best_metric, self.eval(ind, bypass_check= True) - ind.best_metric
        self.task.trainer.update_lr(lr)