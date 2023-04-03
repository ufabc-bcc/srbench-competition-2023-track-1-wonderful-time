from ...components.tree import Tree
from ..population import Individual
from typing import List
import numpy as np
import random
from ...utils.functional import numba_randomchoice_w_prob, softmax

class Crossover:
    def __init__(self, finetune_steps: int = 0, *args, **kwargs):
        self.finetune_steps:int = finetune_steps

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> List[Individual]:
        pass

    def update_task_info(self, **kwargs):
        pass

class SubTreeCrossover(Crossover):
    def __init__(self, finetune_steps: int = 0, *args, **kwargs):
        '''
        finetune step (only new insert branch)
        '''
        super().__init__(finetune_steps, *args, **kwargs)
        
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
        
        
        tar_depth = pa.genes.nodes[tar_point].depth
        tar_level = pa.genes.depth - tar_depth
        max_depth = self.max_depth - tar_level
        
        tar_length = pa.genes.nodes[tar_point].length
        tar_remain_length = pa.genes.length - tar_length
        max_length = self.max_length - tar_remain_length
        
        candidates = []
        candidates_weight = []
        for node in pb.genes.nodes:
            if node.depth <= max_depth and node.length <= max_length:
                candidates.append(node.id)
                candidates_weight.append(node.value)
                
        assert len(candidates)
                        
        #the more weight the more possible of valuable information
        #also want it to have the same sign with the target
        candidates_weight = softmax(np.array(candidates_weight) * np.sign(pa.genes.nodes[tar_point].value))
        
        src_point = candidates[
            numba_randomchoice_w_prob(
                candidates_weight / np.sum(candidates_weight)
            )
        ]
        
        #split tree
        tar_branch, tar_root = pa.genes.split_tree(tar_point)
        src_branch, src_root = pb.genes.split_tree(src_point)
        
        
        #add mask to fine tune branch 
        mask = np.arange(len(tar_root[0]), len(tar_root[0]) + len(src_branch))
        
        
        child_a = Individual(Tree(tar_root[0] + src_branch + tar_root[1], mask = mask
                                  , deep_copoy= True), task = pa.task)
        
        
        assert child_a.genes.length <= self.max_length and child_a.genes.depth <= self.max_depth
        
        #fine-tune the branch
        pa.task.train(child_a, steps = self.finetune_steps)
        
        child_a.genes.remove_mask()
        
        
        children = [child_a]
        
        return children 
        
    
    def update_task_info(self, **kwargs):
        self.max_depth: int = kwargs['max_depth']
        self.max_length: int = kwargs['max_length']