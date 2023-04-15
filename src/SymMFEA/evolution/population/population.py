from ...components.tree import TreeFactory
from .individual import Individual
from ...utils.functional import numba_v2v_int_wrapper
import numpy as np
from ...components.multiprocessor import Multiprocessor

from typing import List
from ...utils.functional import numba_randomchoice
from ...utils.timer import *
from ..task import Task, SubTask
import random

        
class SubPopulation:
        
    def __init__(self,
                 num_inds: int,
                 tree_config: dict = {},
                 skill_factor:int = None, 
                 task: Task = None):
        
        self.skill_factor = skill_factor
        
        
        self.task = SubTask(task)
        
        self.tree_factory = TreeFactory(task_idx = skill_factor, num_total_terminals= len(task.terminal_set), tree_config= tree_config)
        
        self.ls_inds = [Individual(self.tree_factory.create_tree(), task = self.task, skill_factor = skill_factor) for _ in range(num_inds)]
        
        self.scalar_fitness: np.ndarray = None
        self.objective: np.ndarray = None
        
        self.optimized_idx:int = 0
        
    def update_optimized_idx(self):
        self.optimized_idx = len(self.ls_inds)
                
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

    def select(self, index_selected_inds: list):
        self.ls_inds = [self.ls_inds[idx] for idx in index_selected_inds]
        self.update_optimized_idx()
        
    def collect_optimize_jobs(self):
        return [(self.task, ind) for ind in self.ls_inds[self.optimized_idx:]]
            
    def collect_fitness_info(self):
        self.objective = np.array([ind.objective for ind in self.ls_inds])
        self.scalar_fitness = np.mean(self.objective, axis = 1) + 1e-12 
    
    def collect_best_info(self):
        self.best_idx = np.argmax([ind.objective[0] for ind in self.ls_inds])
        self.max_main_objective = self.ls_inds[self.best_idx].objective[0]


# ==================================================================================

class Population:
    def __init__(self, nb_inds_tasks: List[int], task:Task, multiprocessor: Multiprocessor, num_sub_tasks: int = 1, 
        tree_config:dict = {}, offspring_size:float = 1.0) -> None:
        '''
        A Population include:\n
        + `nb_inds_tasks`: number individual of tasks; nb_inds_tasks[i] = num individual of task i
        '''
        # save params
        self.num_sub_tasks = num_sub_tasks
        
        self.nb_inds_tasks = [nb_inds_tasks] * self.num_sub_tasks if isinstance(nb_inds_tasks, int) else nb_inds_tasks
            
        self.ls_subPop: List[SubPopulation] = [
            SubPopulation(self.nb_inds_tasks[skf], skill_factor = skf,  task= task, tree_config = tree_config) for skf in range(self.num_sub_tasks)
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
        
    def collect_optimize_jobs(self):
        optimize_jobs = []
        for subpop in self: 
            optimize_jobs.extend(subpop.collect_optimize_jobs())
            
        #shuffle because some jobs are shorter
        random.shuffle(optimize_jobs)
        return optimize_jobs
        
    def optimize(self):
        optimize_jobs = self.collect_optimize_jobs()

        metrics, loss, train_steps = self.multiprocessor.execute(optimize_jobs)
        
        for metric, job in zip(metrics, optimize_jobs):
            task, ind= job
            ind.objective = [metric if task.is_larger_better else -metric]
            ind.is_optimized = True
        
        self.train_steps += sum(train_steps)
        self.update_optimized_idx()
    
    def update_optimized_idx(self):
        for subpop in self:
            subpop.update_optimized_idx()

        
    @timed
    def collect_fitness_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_fitness_info()
    
    @timed     
    def collect_best_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_best_info()
    
    def get_best_trees(self):
        best_trees = [subPop.ls_inds[subPop.best_idx] for subPop in self]
        
        for tree in best_trees:
            tree.rollback_best()
        
        best_tree = best_trees[np.argmax([subPop.max_main_objective for subPop in self])]
        return best_trees, best_tree