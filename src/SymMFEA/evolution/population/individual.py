import numpy as np
from ...components.tree import Tree
from typing import List
from ..task import SubTask
from ...components.metrics import Metric
import os
class Individual:
    '''
    a Individual include:\n
    + `genes`: Tree\n
    + `skill_factor`: skill factor of the individual\n
    + `fcost`: factorial cost of the individual for skill_factor
    '''
    def __init__(self, genes, task:SubTask, skill_factor: int, *args, **kwargs): 
        self.skill_factor: int = skill_factor
        self.objective: List[float] = None
        self.genes: Tree = genes
        self.best_metric: float = None
        self.nb_consecutive_not_improve: int = 0
        self.task = task
        self.parent_profile: dict= dict()
        self.optimizer_profile: dict= dict()
        self.is_optimized = False
    
    def flush(self):
        self.is_optimized = False
        self.best_metric = None
        self.nb_consecutive_not_improve = 0
    
    @property
    def position(self):
        return self.genes.position
    
    @property
    def main_objective(self):
        return self.objective[0]
        
    def update_best_tree(self, metric):
        self.best_metric = metric
        self.genes.update_best_tree()
    
    def rollback_best(self):
        self.genes.rollback_best()

    def __call__(self, X: np.ndarray, update_stats = False):
        return self.genes(X, update_stats = update_stats)
    
    
    def update_parent_profile(self, **profile):
        for k, v in profile.items():
            self.parent_profile[k] = v
    
    def finetune(self, finetune_steps: int, decay_lr: float):
        self.task.finetune(self, finetune_steps= finetune_steps, decay_lr = decay_lr)
        
    def set_objective(self, objective: np.ndarray, compact:bool = False):
        
        self.objective = [objective]
        
        if compact:
            self.objective.extend([-max(self.genes.length, 10), -max(self.genes.depth, 3)])
        
        
    def run_check(self, metric: Metric):
        if os.environ['DEBUG']:
            met = metric(self.task.data.y_val, self(self.task.data.X_val))
                        
            assert abs((met - self.best_metric) / (self.best_metric + 1e-20)) < 1e-3, (met, self.best_metric) 