from .grad_optimizer import GradOpimizer
from ..data_pool import TrainDataLoader, DataView
from .loss import *
from ..metrics import Metric
from typing import Tuple
from ...utils.progress_bar import FinetuneProgressBar
from tqdm.asyncio import tqdm_asyncio

class Trainer:
    def __init__(self, optimizer: GradOpimizer, loss: Loss, metric: Metric, early_stopping:int = 2, *args, **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.early_stopping = early_stopping
        
    def update_lr(self, lr):
        self.optimizer.update_lr(lr)
        
    def fit(self, ind, data: TrainDataLoader, val_data: DataView, steps: int = 10, finetuner: Tuple[FinetuneProgressBar, tqdm_asyncio]= None):
        if steps == 0:
            return 0
        
        assert not ind.is_optimized
        profile = ind.optimizer_profile
        
        if finetuner is not None: 
            progress, pbar = finetuner
        
        else:
            progress = None
            pbar  = range(steps)
        
        for step in pbar:
            step_loss = []
            while data.hasNext:
                X, y = next(data)
                y_hat = ind(X)
                dY, loss = self.loss(y, y_hat)
                self.optimizer.backprop(ind.genes, dY, profile= profile)
                step_loss.append(loss)
                
            y_hat = ind(val_data.X_val)
            metric = self.metric(val_data.y_val, y_hat)
            
            self.update_learning_state(ind, metric= metric)
            
            if progress is not None: 
                progress.update(loss= loss, metric = metric, best_metric = ind.best_metric, reverse = not self.metric.is_larger_better)
            
            elif ind.nb_consecutive_not_improve == self.early_stopping:
                break
        
        ind.rollback_best()
        
        #update stats
        ind.flush_stats()
        ind(X, update_stats= True)
        
        
        n_test = 0
        while not ind.run_check(self.metric, raise_error = False):
            #update stats multiple time because of batchnorm
            ind(X, update_stats= True)
            n_test += 1
            if n_test > ind.genes.depth:
                raise ValueError('Rollback false!!!!!!!!')
        
        
        
        return ind.best_metric, np.mean(step_loss), step + 1, ind.optimizer_profile 
        
    def update_learning_state(self, ind, metric: float):
        if ind.best_metric is not None: 
            if self.metric.is_better(metric, ind.best_metric):
                ind.update_best_tree(metric)
            else:
                ind.nb_consecutive_not_improve += 1
        else:
            ind.update_best_tree(metric)