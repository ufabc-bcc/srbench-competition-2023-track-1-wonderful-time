from ...components.tree import Tree, get_possible_range
from ..population import Individual
from typing import List
import numpy as np
import random
from ...utils.functional import numba_randomchoice_w_prob, softmax

class Crossover:
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> List[Individual]:
        ...

    def update_population_info(self, **kwargs):
        ...
    
    @staticmethod
    def update_parent_profile(child, parent: List[Individual]):
        child.update_parent_profile(
            born_way= 'crossover',
            num_parents = len(parent),
            parent_objective= [p.objective for p in parent],
            parent_skf = [p.skill_factor for p in parent]
        )

class SubTreeCrossover(Crossover):
    def __init__(self, *args, **kwargs):
        '''
        finetune step (only new insert branch)
        '''
        super().__init__(*args, **kwargs)
        
    def __call__(self, pa: Individual, pb: Individual, *args, **kwargs) -> List[Individual]:
        '''
        replace a branch from pa with a branch from pb  
        '''
        #select cut point on target tree, make sure child not deeper or longer than max
        if pa.genes.length < 2:
            return []
        elif pa.genes.length == 2:
            tar_point = 1
        else:
            tar_point = random.randint(1, pa.genes.length - 1)
        
        
        
        
        max_length, max_depth = get_possible_range(tree= pa, 
                                                   point = tar_point,
                                                   max_depth= self.max_depth[pa.skill_factor],
                                                   max_length= self.max_length[pa.skill_factor])
        
        
        candidates = []
        candidates_weight = []
        for node in pb.genes.nodes:
            if node.depth <= max_depth and node.length <= max_length:
                candidates.append(node.id)
                candidates_weight.append(node.length)
                
        if len(candidates) == 0:
            return []
                        
        #want it as long as possible
        #also want it to have the same sign with the target
        candidates_weight = softmax(np.array(candidates_weight, dtype = np.float64) * np.sign(pa.genes.nodes[tar_point].value))
        
        src_point = candidates[
            numba_randomchoice_w_prob(
                candidates_weight / np.sum(candidates_weight)
            )
        ]
        
        #split tree
        tar_branch, tar_root = pa.genes.split_tree(tar_point)
        src_branch, src_root = pb.genes.split_tree(src_point)
        
        
        
        
        child_a = Individual(Tree(tar_root[0] + src_branch + tar_root[1]
                                  , deepcopy= True), task = pa.task, skill_factor= pa.skill_factor)
        
        
        assert child_a.genes.length <= self.max_length[pa.skill_factor], (child_a.genes.length, self.max_length[pa.skill_factor])
        assert child_a.genes.depth <= self.max_depth[pa.skill_factor], (child_a.genes.depth, self.max_depth[pa.skill_factor])
                
        self.update_parent_profile(child_a, [pa, pb])
        
        children = [child_a]
        
        return children 
        
    
    def update_population_info(self, **kwargs):
        self.max_depth: List[int] = kwargs['max_depth']
        self.max_length: List[int] = kwargs['max_length']