import numpy as np
import numba as nb

@nb.njit(nb.int64[:](nb.float64[:]))
def sort_scalar_fitness(ls_fcost):
    return np.argsort(-ls_fcost)

class Ranker:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def subpop_sort(subpop):
        pass
    
    def __call__(self, population):
        pass

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
    
    def __call__(self, population):
        for subpop in population:
            idx = self.subpop_sort(subpop)
            subpop.ls_inds = [subpop.ls_inds[i] for i in idx]


class NonDominatedRanker:
    pass
