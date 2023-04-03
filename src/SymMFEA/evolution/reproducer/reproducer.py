
from .crossover import *
from .mutation import *
from ..population import Population, SubPopulation
from ...utils.functional import numba_randomchoice
class Reproducer:
    '''
    Reproducer for 1 subpop population
    '''
    def __init__(self, crossover: Crossover, mutation: Mutation, crossover_size: float = 0.5, mutation_size: float = 0.5):
        self.crossover: Crossover = crossover
        self.mutation: Mutation = mutation
        self.crossover_size: float = crossover_size
        self.mutation_size: float = mutation_size
        
    def __call__(self, population: Population):
        
        assert population.num_sub_tasks == 1
        
        cross_offsprings = []
        mutate_offsprings = []
        
        #crossover
        while len(cross_offsprings) < int(len(population) * self.crossover_size):
            # choose parent 
            pa, pb = population.__getRandomInds__(2)
            
            offspingrs = self.crossover(pa, pb)

            cross_offsprings.extend(offspingrs)
            
            
        while len(mutate_offsprings) < int(len(population) * self.crossover_size):
            # choose parent 
            p = population.__getRandomInds__(1)[0]
            
            mutate_offsprings.append(self.mutation(p))

        population.ls_subPop[0].extend(cross_offsprings)
        population.ls_subPop[0].extend(mutate_offsprings)
        
    def update_task_info(self, **kwargs):
        self.crossover.update_task_info(**kwargs)
        self.mutation.update_task_info(**kwargs)