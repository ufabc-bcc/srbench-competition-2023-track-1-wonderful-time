import numpy as np
import numba as nb
from ..utils.timer import *
from pygmo import sort_population_mo as mo_sort

@nb.njit(nb.int64[:](nb.float64[:]))
def sort_scalar_fitness(ls_fcost):
    return np.argsort(-ls_fcost)

class Ranker:
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def subpop_sort(subpop):
        ...
    
    @timed
    def __call__(self, population):
        for subpop in population:
            idx = self.subpop_sort(subpop)
            subpop.ls_inds = [subpop.ls_inds[i] for i in idx]

class SingleObjectiveRanker(Ranker):
    @staticmethod
    def subpop_sort(subpop):
        '''
        Sort a subpopulation
        '''
        if len(subpop.ls_inds):
            idx = sort_scalar_fitness(subpop.scalar_fitness)
        else:
            idx = np.array([])
        
        return idx
    



class NonDominatedRanker(Ranker):
    @staticmethod
    def subpop_sort(subpop):
        '''
        Sort a subpopulation
        '''
        if len(subpop.ls_inds):
            idx = mo_sort(-subpop.objective)            
        else:
            idx = np.array([])
              
        return idx
