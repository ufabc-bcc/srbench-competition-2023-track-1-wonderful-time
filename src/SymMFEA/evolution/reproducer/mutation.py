import numpy as np
from ..population import Individual
from ...components.tree import Tree, FlexTreeFactory, get_possible_range
from ...components.functions import Operand, Node, FUNCTION_SET, LINEAR_FUNCTION_SET
from ...utils.functional import numba_randomchoice_w_prob, normalize_norm1
import random
from typing import List

class Mutation:
    def __init__(self, finetune_steps: int = 0, *args, **kwargs):
        self.finetune_steps:int = finetune_steps

    def __call__(self, parent: Individual) -> Individual:
        pass
    
    def update_task_info(self, **kwargs):
        pass
    
    @staticmethod
    def update_parent_profile(child, parent: Individual):
        child.update_parent_profile(
            born_way= 'mutation',
            num_parents = 1,
            parent_new_born_objective= [parent.new_born_objective],
            parent_skf = [parent.skill_factor]
        )

class VariableMutation(Mutation):
    def __init__(self, *args, **kwargs):
        self.num_total_terminals: int 
    
    def __call__(self, parent: Individual):
        child_nodes = []
        for node in parent.genes.nodes:
            if isinstance(node, Operand):
                child_nodes.append(Operand(index = random.randint(0, self.num_total_terminals - 1)))
            
            else:
                child_nodes.append(Node.deepcopy(node))
        child = Individual(Tree(child_nodes), task= parent.task, skill_factor= parent.skill_factor)
        self.update_parent_profile(child, parent)
        return [child]
    
    def update_task_info(self, **kwargs):
        self.num_total_terminals = kwargs['nb_terminals']


class GrowTreeMutation(Mutation):
    def __init__(self, finetune_steps:int = 0, *args, **kwargs):
        super().__init__(finetune_steps= finetune_steps)
        self.tree_creator: FlexTreeFactory
    
    def __call__(self, parent: Individual):
        assert self.max_length[parent.skill_factor] >= parent.genes.length
        assert self.max_depth[parent.skill_factor] >= parent.genes.depth
        #find leaf able to grow
        select_prob = []
        for i, node in enumerate(parent.genes.nodes):
            r = get_possible_range(tree= parent, 
                                            point = i,
                                            max_depth= self.max_depth[parent.skill_factor],
                                            max_length= self.max_length[parent.skill_factor])
            select_prob.append((r[0] - 1) * (r[1] - 1) * int(node.is_leaf))
        
        select_prob = np.array(select_prob)
        #don't find possible point
        if np.sum(select_prob).item() < 1e-12:
            return []
        
        select_prob = normalize_norm1(select_prob)
        grow_point = numba_randomchoice_w_prob(select_prob)
        
        assert parent.genes.nodes[grow_point].is_leaf
        
        max_length, max_depth = get_possible_range(tree= parent, 
                                            point = grow_point,
                                            max_depth= self.max_depth[parent.skill_factor],
                                            max_length= self.max_length[parent.skill_factor])
        
        self.tree_creator.update_config(
            max_depth= max_depth, max_length= max_length   
        )
        
        #can not create nonlinear branch from a parent nonlinear node
        branch = self.tree_creator.create_tree(root_linear_constrant= not parent.genes.nodes[parent.genes.nodes[grow_point].parent].is_nonlinear).nodes


        mask = np.arange(grow_point, grow_point + len(branch))
        
        
        child = Individual(Tree(parent.genes.nodes[:grow_point] + branch + parent.genes.nodes[grow_point + 1 : ], deepcopy= True, mask= mask), task= parent.task, skill_factor= parent.skill_factor)
        
        assert child.genes.length <= self.max_length[parent.skill_factor], (child.genes.length, self.max_length[parent.skill_factor])
        assert child.genes.depth <= self.max_depth[parent.skill_factor], (child.genes.depth, self.max_depth[parent.skill_factor])
        
        #finetune
        child.finetune(self.finetune_steps, decay_lr= self.finetune_steps)
        child.genes.remove_mask()
        
        self.update_parent_profile(child, parent)
        
        return [child]
    
    def update_task_info(self, **kwargs):
        self.num_total_terminals = kwargs['nb_terminals']
        self.tree_creator = FlexTreeFactory(
            num_total_terminals= self.num_total_terminals,
        )
        self.max_depth: int = kwargs['max_depth']
        self.max_length: int = kwargs['max_length']



#NOTE: Numerically unstable
class NodeMutation(Mutation):
    def __init__(self, *args, **kwargs):
        self.num_total_terminals: int 
    
    def __call__(self, parent: Individual):
        node = parent.genes.nodes[random.randint(0, parent.genes.length - 1)]
        
        funcion_set = FUNCTION_SET if node.is_nonlinear else LINEAR_FUNCTION_SET
        
        candidates = funcion_set[node.arity]
        new_node = Node.deepcopy(node, new_class= candidates[random.randint(0, len(candidates) - 1)])
        
        child =Individual(Tree(parent.genes.nodes[:node.id] + [new_node] + parent.genes.nodes[node.id + 1 : ]), task= parent.task, deepcopy = True, skill_factor= parent.skill_factor)
        self.update_parent_profile(child, parent)
        
        return [child]

        
class MutationList(Mutation):
    '''
    List of mutation to happend with defined probability \n
    Perform one each
    '''
    def __init__(self, mutations: List[Mutation], prob: List[int] = None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if prob is None:
            prob = np.full(len(mutations), 1 / len(mutations))
        else:
            prob = np.array(prob, dtype = np.float64)
        
        assert np.abs(np.sum(prob)) - 1 < 1e-9
        
        self.prob = prob
        self.mutations = mutations
    
    def __call__(self, parent: Individual) -> Individual:
        mut = self.mutations[numba_randomchoice_w_prob(self.prob)]
        return mut(parent)
        
    def update_task_info(self, **kwargs):
        for mut in self.mutations:
            mut.update_task_info(**kwargs)
    