from .grad_optimizer import GradOpimizer
from ..data_pool import DataView
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
        

    def fit(self, ind, data: DataView, steps: int = 10, finetuner: Tuple[FinetuneProgressBar, tqdm_asyncio]= None):
        if steps == 0:
            return {}
        
        assert not ind.is_optimized
        
        
        if finetuner is not None: 
            progress, pbar = finetuner
        
        else:
            progress = None
            pbar  = range(steps)
        
        for step in pbar:
            step_loss = []
            X, y = data.X_train, data.y_train
            y_hat = ind(X, training= True)
            dY, loss = self.loss(y, y_hat)
            ind.optimizer_profile = self.optimizer.backprop(ind.genes, dY, profile= ind.optimizer_profile)
            step_loss.append(loss)
                
            y_hat = ind(data.X_val)
            metric = self.metric(data.y_val, y_hat)
            
            self.update_learning_state(ind, metric= metric)
            
            if progress is not None: 
                progress.update(loss= loss, metric = metric, best_metric = ind.best_metric, reverse = not self.metric.is_larger_better)
            
            elif ind.nb_consecutive_not_improve == self.early_stopping:
                break
        
        ind.rollback_best()
        
        #update stats
        ind(data.X_train, update_stats = True)        
        
        
        #check if rollback successfully
        
        if os.environ.get('DEBUG'):
            met = self.metric(data.y_val, ind(data.X_val))
            
            rs = abs((met - ind.best_metric) / (ind.best_metric + 1e-20)) < 1e-4
            
            
            assert rs, (met, ind.best_metric) 
            
            ind(data.X_train, check_stats= True)
        

        return {
            'best_metric': ind.best_metric,
            'loss': np.mean(step_loss),
            'train_steps': step + 1, 
            'profile': ind.optimizer_profile
        }
        
    def update_learning_state(self, ind, metric: float):
        if self.metric.is_better(metric, ind.best_metric):
            ind.update_best_tree(metric)
        else:
            ind.nb_consecutive_not_improve += 1