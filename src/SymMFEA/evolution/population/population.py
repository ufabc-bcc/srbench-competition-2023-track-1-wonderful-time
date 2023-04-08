from typing import List
from ...components.tree import TreeFactory
from .individual import Individual
from numba import jit 
import numpy as np


from typing import Type, List
from numba import jit
from ...utils.functional import numba_randomchoice

from ..task import Task, SubTask

        
class SubPopulation:
        
    def __init__(self,
                 num_inds: int,
                 tree_config: dict = {},
                 skill_factor:int = None, 
                 task: Task = None):
        
        self.skill_factor = skill_factor
        
        
        self.task = SubTask(task)
        
        self.tree_factory = TreeFactory(terminal_set= self.task.terminal_set,
                                                **tree_config)
        
        self.ls_inds = [Individual(self.tree_factory.create_tree(), task = self.task, skill_factor = skill_factor) for _ in range(num_inds)]
        
        self.scalar_fitness: np.ndarray = None
        self.objective: np.ndarray = None
                
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
    @property 
    def __getWorstIndividual__(self):
        return self.ls_inds[int(np.argmax(self.factorial_rank))]
    
    @staticmethod
    @jit(nopython = True)
    def _numba_meanInds(ls_genes):
        res = [np.mean(ls_genes[:, i]) for i in range(ls_genes.shape[1])]
        return np.array(res)

    @property 
    def __meanInds__(self):
        # return self.__class__._numba_meanInds(np.array([ind.genes for ind in self.ls_inds]))
        return np.mean([ind.genes for ind in self.ls_inds], axis= 0)

    @staticmethod
    @jit(nopython = True)
    def _numba_stdInds(ls_genes):
        res = [np.std(ls_genes[:, i]) for i in range(ls_genes.shape[1])]
        return np.array(res)


    @property 
    def __stdInds__(self):
        # return self.__class__._numba_stdInds(np.array([ind.genes for ind in self.ls_inds]))
        return np.std([ind.genes for ind in self.ls_inds], axis= 0)

    @staticmethod
    @jit(nopython = True)
    def _sort_rank(ls_fcost):
        return np.argsort(np.argsort(ls_fcost)) + 1

    def update_rank(self):
        '''
        Update `factorial_rank` and `scalar_fitness`
        '''
        # self.factorial_rank = np.argsort(np.argsort([ind.objective for ind in self.ls_inds])) + 1
        if len(self.ls_inds):
            self.factorial_rank = self.__class__._sort_rank(np.array([ind.objective for ind in self.ls_inds]))
        else:
            self.factorial_rank = np.array([])
        self.scalar_fitness = 1/self.factorial_rank

    def select(self, index_selected_inds: list):
        self.ls_inds = [self.ls_inds[idx] for idx in index_selected_inds]
        
        
    def optimize(self):
        for ind in self.ls_inds:
            self.task.train(ind)
            metric = self.task.eval(ind)
            ind.objective = [metric if self.task.is_larger_better else -metric]
            
            if ind.new_born_objective is None:
                ind.new_born_objective = ind.objective
            
    def collect_fitness_info(self):
        self.objective = np.array([ind.objective for ind in self.ls_inds])
        self.scalar_fitness = np.mean(self.objective, axis = 1) + 1e-12 
    
    def collect_best_info(self):
        self.best_idx = np.argmax([ind.objective[0] for ind in self.ls_inds])
        self.max_main_objective = self.ls_inds[self.best_idx].objective[0]


# ==================================================================================

class Population:
    def __init__(self, nb_inds_tasks: List[int], task:Task, num_sub_tasks: int = 1, 
        tree_config:dict = {}) -> None:
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

    def optimize(self):
        for supop in self.ls_subPop:
            supop.optimize()
    
    def collect_fitness_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_fitness_info()
            
    def collect_best_info(self):
        for subpop in self.ls_subPop:
            subpop.collect_best_info()
    
    def get_best_trees(self):
        best_trees = [subPop.ls_inds[subPop.best_idx] for subPop in self]
        
        for tree in best_trees:
            tree.rollback_best()
        
        best_tree = best_trees[np.argmax([subPop.max_main_objective for subPop in self])]
        return best_trees, best_tree
        