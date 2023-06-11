from typing import List
from ..components.data_pool import DataView, initDataPool
from ..components.metrics import Metric
import numpy as np
from ..utils import FinetuneProgressBar
from ..utils.timer import *
from ..components import data_pool
from ..components.trainer import Trainer
import multiprocessing as mp


class Task:
    def __init__(self, X: np.ndarray, y:np.ndarray, metric: Metric, test_size: float = 0.2):
        assert len(X.shape) == 2
        self.terminal_set = [i for i in range(X.shape[1])]
        
        #init datapool
        initDataPool(X, y, test_size= test_size)
        self.data_pool = data_pool.data_pool        
        self.is_larger_better = metric.is_larger_better
        self.task_table = mp.Manager().dict()
        self.terminal_table = dict()
    
    def __getitem__(self, idx):
        return self.task_table[idx]
    
    def __setitem__(self, key, value):
        self.task_table[key] = value
    
class SubTask:
    def __init__(self, task: Task, skill_factor: int, data_sample:float = 1, terminal_set:List[int] = None):
        
        self.terminal_set = task.terminal_set if terminal_set is None else terminal_set
        
        self.data = DataView(task.data_pool, data_sample)
        
        self.is_larger_better = task.is_larger_better
        
        self.skill_factor = skill_factor
        task[skill_factor] = self
        task.terminal_table[skill_factor] = self.terminal_set
            
    def finetune(self, ind, trainer: Trainer, finetune_steps: int = 5000, decay_lr: float = 100, compact: bool = True):
        '''
        unlock_view: use all data
        '''
         
        if finetune_steps < 1:
            return
        
        ind.flush()
        self.data.unlock()
        
        
        lr = ind.optimizer_profile.get('lr')
        
        new_lr = lr / decay_lr if lr is not None else trainer.optimizer.lr / decay_lr
        ind.optimizer_profile['lr'] = new_lr
        
        with FinetuneProgressBar(
            num_iters= finetune_steps,
            metric_name= [str(trainer.loss), str(trainer.metric)]
        ) as (progress, pbar):
        
            result = trainer.fit(ind, data = self.data, finetuner= (progress, pbar))
        
        #set objective
        ind.set_objective(result['best_metric'], compact = compact)
        ind.is_optimized = True


