import numpy as np
from ..evolution.population import Population
from ..components import weight_manager 
from typing import List
from ..utils.timer import *

class Selector():
    def __init__(self, select_optimizing_inds: float = 0.5, *args, **kwds) -> None:
        self.select_optimizing_inds = select_optimizing_inds
        
    @timed
    def __call__(self, population:Population, *args, **kwds) -> List[int]:
        ls_idx_selected = []
        for idx_subpop, subpop in enumerate(population):
            N = min(population.nb_inds_tasks[idx_subpop], len(subpop))
            l = len(subpop)
            
            #select top
            idx_selected_inds = np.arange(N).tolist()    
            subpop.select(idx_selected_inds)
            
            #free space not selected idx
            weight_manager.WM.free_space(np.arange(N + 1, l))
            

    
    

