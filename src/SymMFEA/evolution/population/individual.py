import numpy as np
from ...components.tree import Tree
from typing import List
from ..task import SubTask
class Individual:
    '''
    a Individual include:\n
    + `genes`: Tree\n
    + `skill_factor`: skill factor of the individual\n
    + `fcost`: factorial cost of the individual for skill_factor
    '''
    def __init__(self, genes, task:SubTask, *args, **kwargs): 
        self.skill_factor: int = None
        self.objective: List[float] = None
        self.genes: Tree = genes
        self.best_metric: float = None
        self.nb_consecutive_not_improve: int = 0
        self.task = task
    
    @property
    def isPrime(self):
        return self.genes.isPrime
        
    def update_best_tree(self, metric):
        self.best_metric = metric
        self.genes.update()
    
    def rollback_best(self):
        self.genes.rollback_best()

    def __call__(self, X: np.ndarray):
        return self.genes(X)
    
    @property
    def stop_optimize(self):
        return self.task.stop_optimize(self)