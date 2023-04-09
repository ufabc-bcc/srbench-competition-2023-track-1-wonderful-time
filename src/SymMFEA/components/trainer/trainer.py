from .grad_optimizer import GradOpimizer
from ..data_pool import TrainDataLoader
from .loss import *
class Trainer:
    def __init__(self, optimizer: GradOpimizer, loss: Loss, *args, **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        
    def update_lr(self, lr):
        self.optimizer.update_lr(lr)
        
    def fit(self, ind, data: TrainDataLoader, steps: int = 10):
        if steps == 0:
            return 0
        for _ in range(steps):
            step_loss = []
            while data.hasNext:
                X, y = next(data)
                y_hat = ind(X)
                dY, loss = self.loss(y, y_hat)
                self.optimizer.backprop(ind.genes, dY)
                step_loss.append(loss)
        
        return np.mean(step_loss) 
        
            