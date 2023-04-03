from typing import List
from ..components.trainer import Trainer
from ..components.data_pool import DataPool, DataView, TrainDataLoader
from ..components.trainer import Loss, GradOpimizer
from ..components.metrics import Metric
import numpy as np
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
    
    def train(self, ind, steps = None):
        if self.stop_optimize(ind):
            return
            
        self.task.trainer.fit(ind, self.train_dataloader, self.task.steps_per_gen if steps is None else steps)
            
    def eval(self, ind):
        if not self.stop_optimize(ind):
            y_hat = ind(self.data.X_val)
            metric =  self.task.metric(self.data.y_val, y_hat)
            self.update_learning_state(ind, metric)
            
        return ind.best_metric
    
    def update_learning_state(self, ind, metric: float):
        if ind.best_metric is not None: 
            if (metric > ind.best_metric) != self.task.metric.is_larger_better:
                ind.nb_consecutive_not_improve += 1
            else:
                ind.update_best_tree(metric)
        else:
            ind.update_best_tree(metric)
    
    def stop_optimize(self, ind):
        if self.task.nb_not_improve is not None:
            if ind.nb_consecutive_not_improve >= self.task.nb_not_improve:
                if not ind.isPrime:
                    ind.rollback_best()
                return True
        
        return False