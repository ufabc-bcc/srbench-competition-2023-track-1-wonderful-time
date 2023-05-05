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
                 shuffle: bool, test_size: float = 0.2, trainer_config:dict = {}):
        assert len(X.shape) == 2
        self.terminal_set = [i for i in range(X.shape[1])]
        
        #init datapool
        initDataPool(X, y, test_size= test_size)
        self.data_pool = data_pool.data_pool
        self.data = DataView(self.data_pool)
        
        self.trainer = Trainer(loss= loss, optimizer=optimizer, metric= metric, **trainer_config)
        self.train_dataloader_cfg = {
            'batch_size': batch_size,
            'shuffle': shuffle,
        }
        self.steps_per_gen:int = steps_per_gen
        self.is_larger_better = metric.is_larger_better
        
    
class SubTask:
    def __init__(self, task: Task, data_sample:float = 1, terminal_set:List[int] = None):
        
        self.terminal_set = task.terminal_set if terminal_set is None else terminal_set
        self.task = task
        self.data = DataView(task.data_pool, data_sample)
        self.train_dataloader = TrainDataLoader(self.data, **self.task.train_dataloader_cfg)
        self.is_larger_better = self.task.is_larger_better
    
    @timed
    def train(self, ind, steps = None):            
        return self.task.trainer.fit(ind, self.train_dataloader, steps= self.task.steps_per_gen if steps is None else steps, val_data = self.data)
        
    
    def eval(self, ind):
        return ind(self.data.y_val)
    
    @timed
    def finetune(self, ind, finetune_steps: int = 5000, decay_lr: float = 100):
        '''
        unlock_view: use all data
        '''
         
        if finetune_steps < 1:
            return
        
        ind.flush()
        self.train_dataloader.unlock()
        self.data.unlock()
        
        lr= self.task.trainer.optimizer.lr 
        self.task.trainer.update_lr(self.task.trainer.optimizer.lr / decay_lr)
        
        with FinetuneProgressBar(
            num_iters= finetune_steps,
            metric_name= [str(self.task.trainer.loss), str(self.task.trainer.metric)]
        ) as (progress, pbar):
        
            result = self.task.trainer.fit(ind, data = self.train_dataloader, val_data = self.data, finetuner= (progress, pbar))
            
                
                        
        self.task.trainer.update_lr(lr)
        
        return result
