import numpy as np
from ..evolution.population import Population
from math import ceil
from typing import List
from ..utils.timer import *

class Selector():
    def __init__(self, select_optimizing_inds: float = 0.5, *args, **kwds) -> None:
        self.select_optimizing_inds = select_optimizing_inds
        
    def __call__(self, population:Population,*args, **kwds) -> List[int]:
        ...
    
class ElitismSelector(Selector):
    def __init__(self, random_percent = 0, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        assert 0<= random_percent and random_percent <= 1
        self.random_percent = random_percent        
    
    @timed
    def __call__(self, population:Population, *args, **kwds) -> List[int]:
        ls_idx_selected = []
        for idx_subpop, subpop in enumerate(population):
            N_i = min(population.nb_inds_tasks[idx_subpop], len(subpop))

            # on top
            N_elitism = ceil(N_i* (1 - self.random_percent))
            idx_selected_inds = np.arange(N_elitism).tolist()
                
            subpop.select(idx_selected_inds)

            ls_idx_selected.append(idx_selected_inds)

        return ls_idx_selected
    
    

