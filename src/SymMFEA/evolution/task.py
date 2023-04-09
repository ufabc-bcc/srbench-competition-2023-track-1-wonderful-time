from typing import List
from ..components.trainer import Trainer
from ..components.data_pool import DataPool, DataView, TrainDataLoader
from ..components.trainer import Loss, GradOpimizer
from ..components.metrics import Metric
import numpy as np
from ..utils import FinetuneProgressBar
from ..utils.timer import *


class Task:
    def __init__(self, X: np.ndarray, y:np.ndarray, loss: Loss,
                 optimizer: GradOpimizer, metric: Metric, steps_per_gen: int, batch_size:int,
                 shuffle: bool, nb_not_improve: int = None, test_size: float = 0.2):
        assert len(X.shape) == 2
        self.terminal_set = [i for i in range(X.shape[1])]
        self.data_pool = DataPool(X, y, test_size= test_size)
        self.trainer = Trainer(loss= loss, optimizer=optimizer)
        self.metric = metric
        self.train_dataloader_cfg = {
            'batch_size': batch_size,
            'shuffle': shuffle,
        }
        self.steps_per_gen:int = steps_per_gen
        self.nb_not_improve = nb_not_improve
        
    
class SubTask:
    def __init__(self, task: Task, data_sample:float = 1, terminal_set:List[int] = None):
        if terminal_set is None:
            self.terminal_set = task.terminal_set 
        self.task = task
        self.data = DataView(task.data_pool, data_sample)
        self.train_dataloader = TrainDataLoader(self.data, **self.task.train_dataloader_cfg)
        self.is_larger_better = self.task.metric.is_larger_better
    
    @timed
    def train(self, ind, steps = None):
        if self.stop_optimize(ind):
            return
            
        self.task.trainer.fit(ind, self.train_dataloader, self.task.steps_per_gen if steps is None else steps)
    
    @timed  
    def eval(self, ind, bypass_check: bool = False):
        
        if not self.stop_optimize(ind) or bypass_check:
            y_hat = ind(self.data.X_val)
            metric =  self.task.metric(self.data.y_val, y_hat)
            self.update_learning_state(ind, metric)
            
            #get real current metric 
            if bypass_check:
                return metric
            
        return ind.best_metric
    
    def update_learning_state(self, ind, metric: float):
        if ind.best_metric is not None: 
            if (metric > ind.best_metric) != self.task.metric.is_larger_better:
                ind.nb_consecutive_not_improve += 1
            else:
                ind.update_best_tree(metric)
        else:
            ind.update_best_tree(metric)
    
    def stop_optimize(self, ind, nb_not_improve = None):
        nb_not_improve = nb_not_improve if nb_not_improve is not None else self.task.nb_not_improve
        if nb_not_improve is not None:
            if ind.nb_consecutive_not_improve >= nb_not_improve:
                if not ind.isPrime:
                    ind.rollback_best()
                return True
        
        return False
    
    def flush_history(self, ind):
        ind.flush_history()
        
        
    def finetune(self, ind, finetune_steps: int = 5000, decay_lr: float = 100, verbose = False):
        self.flush_history(ind)
        lr= self.task.trainer.optimizer.lr 
        self.task.trainer.update_lr(self.task.trainer.optimizer.lr / decay_lr)
        
        pbar = FinetuneProgressBar(
            num_iters= finetune_steps,
            metric_name= [str(self.task.trainer.loss), str(self.task.metric)]
        ) if verbose else range(finetune_steps)
        
        for step in pbar.pbar if verbose else pbar:
            loss = self.task.trainer.fit(ind, self.train_dataloader, 1)
            metric = self.eval(ind, bypass_check= True)
            self.update_learning_state(ind, metric)
            
            if verbose:
                pbar.update(loss= loss, metric = metric, best_metric = ind.best_metric, reverse = not self.task.metric.is_larger_better)
            
        ind.rollback_best()
        
        # assert self.eval(ind, bypass_check= True) == ind.best_metric, self.eval(ind, bypass_check= True) - ind.best_metric
        self.task.trainer.update_lr(lr)