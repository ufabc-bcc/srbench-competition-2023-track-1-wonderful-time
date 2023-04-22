
from .crossover import *
from .mutation import *
from ..population import Population, SubPopulation
from ...utils.functional import numba_randomchoice
from ...utils.timer import *

#NOTE: this reproducer call is outdated
class Reproducer:
    '''
    Reproducer for 1 subpop population
    '''
    def __init__(self, crossover: Crossover, mutation: Mutation, crossover_size: float = 0.5, mutation_size: float = 0.5, **params):
        self.crossover: Crossover = crossover
        self.mutation: Mutation = mutation
        self.crossover_size: float = crossover_size
        self.mutation_size: float = mutation_size
    
    
    def select_crossover_parent(population: Population):
        return population.__getRandomInds__(2)
    
    def select_mutation_parent(population: Population):
        return population.__getRandomInds__(1)[0]
    
    @timed
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
    
    def update_smp(self, Delta_task, count_Delta_tasks):
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
    def __init__(self, crossover: Crossover, mutation: Mutation, **params):
        super().__init__(crossover, mutation, **params)
        self.smp: battle_smp
        
    
    def select_mutation_parent(self, population: Population, skf: int):
        return population.ls_subPop[skf].__getRandomItems__(2, False)
    
    
    def select_crossover_parent(self, population: Population):
        #choose the first parent subpop
        skf_pa = numba_randomchoice_w_prob(self.p_choose_father)
        smp = self.smp[skf_pa].get_smp()
        #choose the second parent subpop
        skf_pb = numba_randomchoice_w_prob(smp)
        
        if skf_pb < population.num_sub_tasks:
            if skf_pa == skf_pb:
                pa, pb = population[skf_pa].__getRandomItems__(size= 2, replace=False)
            else:
                pa = population[skf_pa].__getRandomItems__()
                pb = population[skf_pb].__getRandomItems__()
                
            return [pa, pb]
        
        else: return skf_pa
        
    
    
    def update_task_info(self, **kwargs):
        super().update_task_info(**kwargs)
        self.smp = kwargs['smp']
        self.p_choose_father = kwargs['p_choose_father']
        
    @timed
    def __call__(self, population: Population):              
        num_offsprings = 0
        total_num_offsprings = sum(population.nb_inds_tasks) * population.offspring_size
        offsprings = [[] for _ in range(population.num_sub_tasks)]
        while num_offsprings  < total_num_offsprings:
            
            
            parents = self.select_crossover_parent(population= population)
            
            #if crossover
            if type(parents) == list:
                new_offsprings= self.crossover(*parents)
                  
            #else mutation
            else:
                parents = self.select_mutation_parent(population= population, skf= parents)
                new_offsprings = []
                for p in parents:
                    new_offsprings.extend(self.mutation(p))
            
            num_offsprings+= len(new_offsprings)   
        
            #append offsprings    
            offsprings[parents[0].skill_factor].extend(new_offsprings)
        
        for subpop, off in zip(population, offsprings):
            #remove unknown operands
            for o in off:
                subpop.tree_factory.convert_tree(o.genes)
                
            
        all_offsprings = []
        for o in offsprings:
            all_offsprings.extend(o)
            
        return all_offsprings
    
    @timed    
    def update_smp(self, population: Population, offsprings: List[Individual]):
        Delta:List[List[float]] = np.zeros((population.num_sub_tasks, population.num_sub_tasks + 1)).tolist()
        count_Delta: List[List[float]] = np.zeros((population.num_sub_tasks, population.num_sub_tasks + 1)).tolist()  
        num_task = population.num_sub_tasks
        
        for skf, offsprings_skf in enumerate(offsprings):
            for o in offsprings_skf:
                num_parents = o.parent_profile.get('num_parents')
                for i in range(num_parents):
                    parent_objective = o.parent_profile.get('parent_objective')[i]
                    
                    
                    
                    #NOTE: hardcode first objective
                    d = (o.main_objective - parent_objective[0]) / (parent_objective[0] ** 2 + 1e-50)

                    if o.parent_profile.get('born_way') == 'crossover':
                        Delta[skf][o.parent_profile.get('parent_skf')[i]] += max([d, 0])**2
                        count_Delta[skf][o.parent_profile.get('parent_skf')[i]] += 1
                    else:
                        Delta[skf][num_task] += max([d, 0])**2
                        count_Delta[skf][num_task] += 1
                    

        for i, smp in enumerate(self.smp):
            smp.update_smp(Delta_task= Delta[i], count_Delta_tasks= count_Delta[i])