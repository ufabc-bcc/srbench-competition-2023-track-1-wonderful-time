import numpy as np
from ...components.tree import Tree
from typing import List
from ..task import SubTask
from .. import task
from ...components.metrics import Metric
import os

MIN_AGE = 5
class Individual:
    '''
    a Individual include:\n
    + `genes`: Tree\n
    + `skill_factor`: skill factor of the individual\n
    '''
    __slots__ = [
        'skill_factor',
        'objective',
        'genes',
        'best_metric',
        'nb_consecutive_not_improve',
        'parent_profile',
        'is_optimized',
        'age',
        'optimizer_profile',
    ]
    def __init__(self, genes, skill_factor: int, *args, **kwargs): 
        self.skill_factor: int = skill_factor
        self.objective: List[float] = None
        self.genes: Tree = genes
        self.best_metric: float = None
        self.nb_consecutive_not_improve: int = 0
        self.parent_profile: dict= dict()
        self.optimizer_profile: dict= dict()
        self.is_optimized = False
        self.age: int = 0
    
    def flush(self):
        self.is_optimized = False
        self.best_metric = None
        self.nb_consecutive_not_improve = 0
        self.objective = None
        self.genes.flush()
    
    @property
    def position(self):
        return self.genes.position
    
    @property
    def main_objective(self):
        '''
        main_objective: larger is better while best metric is what it should be
        '''
        return self.objective[0]
        
    def update_best_tree(self, metric):
        self.best_metric = metric
        self.genes.update_best_tree()
    
    def rollback_best(self):
        self.genes.rollback_best()

    def __call__(self, X: np.ndarray, update_stats= False, training= False, check_stats= False):
        return self.genes(X, update_stats= update_stats, training= training, check_stats = check_stats)
    
    # def update_stats(self):
    #     self(self.task.data.X_train, update_stats= True)
        
    def free_space(self):
        self.genes.free_space()
        
    # def run_check_stats(self):
    #     if os.environ.get('DEBUG'):
    #         self(self.task.data.X_train, check_stats= True)
    
    def update_parent_profile(self, **profile):
        for k, v in profile.items():
            self.parent_profile[k] = v
        
    def set_objective(self, *objective: np.ndarray, compact:bool = False, age: bool = True):
        
        self.objective = [*objective]
        
        if compact:
            self.objective.extend([-max(self.genes.length, 10), -max(self.genes.depth, 3), -self.genes.num_nonlinear])
        
        if age:
            self.set_age_objectie(append = True)
            
        
    def set_age_objectie(self, append = False): 
        age = min(MIN_AGE, self.age)
        if append:
            self.objective.append(age)
        else:
            self.objective[-1] = age
        
            
    @property
    def better_than_parent(self) -> bool:
        num_parents = self.parent_profile.get('num_parents')
        
        for i in range(num_parents):
            parent_objective = self.parent_profile.get('parent_objective')[i]
            parent_skf = self.parent_profile.get('parent_skf')[i]
            
            if parent_skf == self.skill_factor:
                #NOTE: HARD CODE Tolerance here
                if self.main_objective - parent_objective[0] < 1e-3:
                    return False
        
        return True
        
    # def run_check(self, metric: Metric):
    #     if os.environ.get('DEBUG'):
    #         met = metric(self.task.data.y_val, self(self.task.data.X_val))
            
    #         rs = abs((met - self.best_metric) / (self.best_metric + 1e-20)) < 1e-4
            
            
    #         assert rs, (met, self.best_metric) 
    

             
            
    def scale(self, scale_factor: float):
        self.genes.scale(scale_factor= scale_factor)