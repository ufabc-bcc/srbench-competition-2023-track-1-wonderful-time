
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
    
    
    @staticmethod
    def select_crossover_parent(population: Population):
        return population.__getRandomInds__(2)
    
    @staticmethod
    def select_mutation_parent(population: Population):
        return population.__getRandomInds__(1)[0]
    

    def __call__(self, population: Population):
        
        assert population.num_sub_tasks == 1
        
        cross_offsprings = []
        mutate_offsprings = []
        
        #crossover
        while len(cross_offsprings) < int(len(population) * self.crossover_size):
            # choose parent 
            
            pa, pb = self.select_crossover_parent(population)
            
            offspingrs = self.crossover(pa, pb)

            cross_offsprings.extend(offspingrs)
            
            
        while len(mutate_offsprings) < int(len(population) * self.crossover_size):
            # choose parent 
            p = self.select_mutation_parent(population)
            
            mutate_offsprings.extend(self.mutation(p))

        population.ls_subPop[0].extend(cross_offsprings)
        population.ls_subPop[0].extend(mutate_offsprings)
        
    def update_task_info(self, **kwargs):
        self.crossover.update_task_info(**kwargs)
        self.mutation.update_task_info(**kwargs)
        
        
class battle_smp:
    def __init__(self, idx_host: int, nb_tasks: int, lr, p_const_intra) -> None:
        assert idx_host < nb_tasks
        self.idx_host = idx_host
        self.nb_tasks = nb_tasks

        #value const for intra
        self.p_const_intra = p_const_intra
        self.lower_p = 0.1/(self.nb_tasks + 1)

        # smp without const_val of host
        self.sum_not_host = 1 - 0.1 - p_const_intra
        self.SMP_not_host: np.ndarray = ((np.zeros((nb_tasks + 1, )) + self.sum_not_host)/(nb_tasks + 1))
        self.SMP_not_host[self.idx_host] += self.sum_not_host - np.sum(self.SMP_not_host)

        smp_return : np.ndarray = np.copy(self.SMP_not_host)
        smp_return[self.idx_host] += self.p_const_intra
        smp_return += self.lower_p

        self.SMP_include_host = smp_return
        self.lr = lr

    def get_smp(self) -> np.ndarray:
        return np.copy(self.SMP_include_host)
    
    def update_SMP(self, Delta_task, count_Delta_tasks):
        '''
        Delta_task > 0 
        '''

        if np.sum(Delta_task) != 0:         
            # newSMP = np.array(Delta_task) / (self.SMP_include_host)
            newSMP = (np.array(Delta_task) / (np.array(count_Delta_tasks) + 1e-50))
            newSMP = newSMP / (np.sum(newSMP) / self.sum_not_host + 1e-50)

            self.SMP_not_host = self.SMP_not_host * (1 - self.lr) + newSMP * self.lr
            
            self.SMP_not_host[self.idx_host] += self.sum_not_host - np.sum(self.SMP_not_host)
            
            smp_return : np.ndarray = np.copy(self.SMP_not_host)
            smp_return[self.idx_host] += self.p_const_intra
            smp_return += self.lower_p
            self.SMP_include_host = smp_return

        return self.SMP_include_host

class SMP_Reproducer(Reproducer):
    def __init__(self, crossover: Crossover, mutation: Mutation, crossover_size: float = 0.5, mutation_size: float = 0.5):
        super().__init__(crossover, mutation, crossover_size, mutation_size)
        self.smp: battle_smp
    
    @staticmethod
    def select_crossover_parent(population: Population):
        return population.__getRandomInds__(2)
    
    
    def update_task_info(self, **kwargs):
        super().update_task_info(**kwargs)
        