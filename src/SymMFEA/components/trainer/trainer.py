from .grad_optimizer import GradOpimizer
from ..data_pool import TrainDataLoader, DataView
from .loss import *
from ..metrics import Metric
class Trainer:
    def __init__(self, optimizer: GradOpimizer, loss: Loss, metric: Metric, *args, **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        
    def update_lr(self, lr):
        self.optimizer.update_lr(lr)
        
    def fit(self, ind, data: TrainDataLoader, val_data: DataView, steps: int = 10, early_stopping:int = 2):
        if steps == 0:
            return 0
        
        assert not ind.is_optimized
        profile = ind.optimizer_profile
        
        for _ in range(steps):
            step_loss = []
            while data.hasNext:
                X, y = next(data)
                y_hat = ind(X, update_stats = True)
                dY, loss = self.loss(y, y_hat)
                self.optimizer.backprop(ind.genes, dY, profile= profile)
                step_loss.append(loss)
                
            y_hat = ind(val_data.X_val)
            metric = self.metric(val_data.y_val, y_hat)
            
            self.update_learning_state(ind, metric= metric)
            
            if ind.nb_consecutive_not_improve == early_stopping:
                ind.rollback_best()
                break
        
        ind.is_optimized = True
        return ind.best_metric, np.mean(step_loss) 
        
    def update_learning_state(self, ind, metric: float):
        if ind.best_metric is not None: 
            if (metric > ind.best_metric) != self.metric.is_larger_better:
                ind.nb_consecutive_not_improve += 1
            else:
                ind.update_best_tree(metric)
        else:
            ind.update_best_tree(metric)