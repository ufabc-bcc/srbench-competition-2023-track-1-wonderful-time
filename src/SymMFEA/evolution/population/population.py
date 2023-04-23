from ...components.tree import TreeFactory
from .individual import Individual
from ...utils.functional import numba_v2v_int_wrapper
import numpy as np
from ...components.multiprocessor import Multiprocessor
from pygmo import fast_non_dominated_sorting
from typing import List
from ...utils.functional import numba_randomchoice
from ...utils.timer import *
from ..task import Task, SubTask
import random
from .. import offsprings_pool

        
class SubPopulation:
        
    def __init__(self,
                 num_inds: int,
                 data_samples: float,
                 tree_config: dict = {},
                 skill_factor:int = None, 
                 task: Task = None):
        
        self.skill_factor = skill_factor
        
        
        self.task = SubTask(task, data_sample= data_samples)
        
        self.tree_factory = TreeFactory(task_idx = skill_factor, num_total_terminals= len(task.terminal_set), tree_config= tree_config)
        
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


    @staticmethod
    @numba_v2v_int_wrapper
    def _sort_rank(ls_fcost):
        return np.argsort(np.argsort(ls_fcost)) + 1

    def update_rank(self):
        '''
        Update `factorial_rank` and `scalar_fitness`
        '''
        if len(self.ls_inds):
            self.factorial_rank = self.__class__._sort_rank(np.array([ind.objective for ind in self.ls_inds]))
        else:
            self.factorial_rank = np.array([])
        self.scalar_fitness = 1/self.factorial_rank
        
    def append(self, offsprings: List[Individual]):
        self.ls_inds.extend(offsprings)
        # self.update_optimized_idx

    def select(self, index_selected_inds: list):
        self.ls_inds = [self.ls_inds[idx] for idx in index_selected_inds]
        # self.update_optimized_idx()
        
            
    def collect_fitness_info(self):
        self.objective = np.array([ind.objective for ind in self.ls_inds])
        self.scalar_fitness = np.mean(self.objective, axis = 1) + 1e-12 
    
    def collect_best_info(self):
        self.best_idx = np.argmax([ind.main_objective for ind in self.ls_inds])
        self.max_main_objective = self.ls_inds[self.best_idx].main_objective


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
            
        self.ls_subPop: List[SubPopulation] = [
            SubPopulation(self.nb_inds_tasks[skf], self.data_sample[skf], skill_factor = skf,  task= task, tree_config = tree_config) for skf in range(self.num_sub_tasks)
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
    def collect_best_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_best_info()
    
    def get_final_candidates(self):
        '''
        get first front if MOO
        '''
        if self.moo:
            trees = self.all()
            fronts, _, _, _ = fast_non_dominated_sorting([-np.array(tree.objective) for tree in trees])
            
            candidates = [trees[i] for i in fronts[0]]
        
        else:
            candidates = [subPop.ls_inds[subPop.best_idx] for subPop in self]
            
        return candidates