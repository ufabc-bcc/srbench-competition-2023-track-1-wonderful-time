from ...components.tree import TreeFactory
from .individual import Individual
import numpy as np
from ...components.multiprocessor import Multiprocessor
from pygmo import fast_non_dominated_sorting
from typing import List
from ...utils.functional import numba_randomchoice
from ...utils.timer import *
from ..task import Task, SubTask
from ...components.column_sampler import ColumnSampler

        
class SubPopulation:
        
    def __init__(self,
                 num_inds: int,
                 data_samples: float,
                 column_sampler: ColumnSampler,
                 tree_config: dict = {},
                 skill_factor:int = None, 
                 task: Task = None):
        
        self.skill_factor = skill_factor
        
        self.tree_factory = TreeFactory(task_idx = skill_factor, num_total_terminals= len(task.terminal_set), tree_config= tree_config, column_sampler= column_sampler)
        
        self.task = SubTask(task, data_sample= data_samples, terminal_set= self.tree_factory.terminal_set)
        
        self.ls_inds = [Individual(self.tree_factory.create_tree(), task = self.task, skill_factor = skill_factor) for _ in range(num_inds)]
        
        self.scalar_fitness: np.ndarray = None
        self.objective: np.ndarray = None
        
        self.optimized_idx:int = 0
                
    def __len__(self): 
        return len(self.ls_inds)
    
    def __getRandomItems__(self, size:int = None, replace:bool = False):
        if size == 0:
            return []
        if size == None:
            return self.ls_inds[numba_randomchoice(len(self), size= None, replace= replace)]
        return [self.ls_inds[idx] for idx in numba_randomchoice(len(self), size= size, replace= replace).tolist()]


    def extend(self, new_inds: List[Individual]):
        self.ls_inds.extend(new_inds)

    def __add__(self, other):
        assert self.task == other.task, 'Cannot add 2 sub-population do not have the same task'
        UnionSubPop = SubPopulation(
            IndClass= self.IndClass,
            skill_factor = self.skill_factor,
            dim= self.dim,
            num_inds= 0,
            task= self.task
        )
        UnionSubPop.ls_inds = self.ls_inds + other.ls_inds
        UnionSubPop.update_rank()
        return UnionSubPop

    @property 
    def __getBestIndividual__(self):
        return self.ls_inds[int(np.argmin(self.factorial_rank))]   

        
    def append(self, offsprings: List[Individual]):
        self.ls_inds.extend(offsprings)

    def select(self, index_selected_inds: list):
        self.ls_inds = [self.ls_inds[idx] for idx in index_selected_inds]
        
    def update_age(self):
        for ind in self.ls_inds:
            ind.age += 1
            
            
    def collect_fitness_info(self):
        self.objective = np.array([ind.objective for ind in self.ls_inds])
        self.main_objective = np.array([ind.main_objective for ind in self.ls_inds])
        self.scalar_fitness = np.mean(self.objective, axis = 1) + 1e-12 
    
    def collect_best_info(self):
        self.best_idx = np.argmax(self.main_objective)
        self.max_main_objective = self.main_objective[self.best_idx]


# ==================================================================================

class Population:
    def __init__(self, nb_inds_tasks: List[int], task:Task, 
                 multiprocessor: Multiprocessor, num_sub_tasks: int = 1, moo:bool= False,
                 data_sample:float = 0.8, tree_config:dict = {}, offspring_size:float = 1.0) -> None:
        '''
        A Population include:\n
        + `nb_inds_tasks`: number individual of tasks; nb_inds_tasks[i] = num individual of task i
        '''
        # save params
        self.num_sub_tasks = num_sub_tasks
        self.data_sample = data_sample
        self.nb_inds_tasks = nb_inds_tasks
        self.moo= moo
        
        
        self.column_sampler = ColumnSampler()
        self.column_sampler.fit(task.data.X_train)
        
        self.ls_subPop: List[SubPopulation] = [
            SubPopulation(self.nb_inds_tasks[skf], self.data_sample[skf], skill_factor = skf,  task= task, tree_config = tree_config, column_sampler= self.column_sampler) for skf in range(self.num_sub_tasks)
        ]
        self.offspring_size= offspring_size
        
        self.train_steps:int = 0
        self.multiprocessor = multiprocessor
        
    def update_nb_inds_tasks(self, nb_inds_tasks):
        self.nb_inds_tasks = nb_inds_tasks

    def __len__(self):
        return sum([len(subPop) for subPop in self.ls_subPop])

    def __getitem__(self, index) -> SubPopulation: 
        return self.ls_subPop[index]
    
    def all(self):
        inds = []
        for subpop in self:
            inds.extend(subpop.ls_inds)
        return inds

    def __getRandomInds__(self, size: int = None, replace: bool = False):
        if size == None:
            return self.ls_subPop[np.random.randint(0, self.num_sub_tasks)].__getRandomItems__(None, replace) 
        else:
            nb_randInds = [0] * self.num_sub_tasks
            for idx in numba_randomchoice(self.num_sub_tasks, size = size, replace= True).tolist():
                nb_randInds[idx] += 1

            res = []
            for idx, nb_inds in enumerate(nb_randInds):
                res += self.ls_subPop[idx].__getRandomItems__(size = nb_inds, replace= replace)

            return res
        
    @timed
    def collect_fitness_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_fitness_info()
    
    @timed
    def update_age(self):
        for subpop in self.ls_subPop:
            subpop.update_age()
            
    @timed     
    def collect_best_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_best_info()
    
    def get_final_candidates(self, min_candidates: int):
        '''
        get first front if MOO
        '''
        if self.moo:
            trees = self.all()
            
            
            fronts, _, _, _ = fast_non_dominated_sorting(-np.array([tree.objective for tree in trees]))
            
            
            min_candidates = max(min_candidates, len(fronts[0]))
            f = 0
            candidates = []
            while len(candidates)  < min_candidates:
                nb_to_take = min_candidates - len(candidates)
                candidates.extend([trees[i] for i in fronts[f][:nb_to_take]])
                f+=1
                
        
        else:
            candidates = [subPop.ls_inds[subPop.best_idx] for subPop in self]
            
        return candidates
    
    def extend(self, offsprings: List[List[Individual]]):
        
        for subpop, offspring in zip(self, offsprings):
            #filter offspring not better than parent
            betters = []
            for o in offspring:
                if o.better_than_parent:
                    betters.append(o)
                else:
                    #free space
                    o.free_space()
                    
            subpop.ls_inds.extend(betters)