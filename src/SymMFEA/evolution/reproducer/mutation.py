import numpy as np
from ..population import Individual
from ...components.tree import Tree, FlexTreeFactory, get_possible_range
from ...components.functions import Operand, Node, FUNCTION_SET, LINEAR_FUNCTION_SET, Constant, Percentile
from ...utils.functional import numba_randomchoice_w_prob, normalize_norm1, numba_randomchoice
import random
from typing import List

class Mutation:
    def __init__(self, *args, **kwargs):
        ...
        
    def __call__(self, parent: Individual) -> Individual:
        ...
    
    def update_population_info(self, **kwargs):
        ...
    
    @staticmethod
    def update_parent_profile(child, parent: Individual):
        child.update_parent_profile(
            born_way= 'mutation',
            num_parents = 1,
            idx_target_smp = 0,
            parent_objective= [parent.objective],
            parent_skf = [parent.skill_factor]
        )

class VariableMutation(Mutation):
    def __init__(self, *args, **kwargs):
        self.num_total_terminals: int 
        self.mutate_variable_size: float = kwargs.get('mutate_variable_size', 0.3)
    
    def __call__(self, parent: Individual):
        child_nodes = []
        
        operand_idx = np.array([i for i in range(parent.genes.length) if isinstance(parent.genes.nodes[i], Operand)], dtype = np.int64)
        
        mutate_idx = numba_randomchoice(operand_idx, size = int(self.mutate_variable_size * len(operand_idx)), replace= False)
        
        for i, node in enumerate(parent.genes.nodes):
            if i in mutate_idx:
                child_nodes.append(Operand(index = random.choice(parent.terminal_set)))
            
            else:
                child_nodes.append(Node.deepcopy(node))
                
        child = Individual(Tree(child_nodes), task= parent.task, skill_factor= parent.skill_factor)
        self.update_parent_profile(child, parent)
        return [child]
    
    def update_population_info(self, **kwargs):
        self.num_total_terminals = kwargs['nb_terminals']


class GrowTreeMutation(Mutation):
    def __init__(self, *args, **kwargs):
        super().__init__()
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
            max_depth= max_depth, max_length= max_length, terminal_set= parent.terminal_set   
        )
        
        #can not create nonlinear branch from a parent nonlinear node
        branch = self.tree_creator.create_tree(root_linear_constrant= not parent.genes.nodes[parent.genes.nodes[grow_point].parent].is_nonlinear).nodes


        
        
        child = Individual(Tree(parent.genes.nodes[:grow_point] + branch + parent.genes.nodes[grow_point + 1 : ], deepcopy= True), task= parent.task, skill_factor= parent.skill_factor)
        
        assert child.genes.length <= self.max_length[parent.skill_factor], (child.genes.length, self.max_length[parent.skill_factor])
        assert child.genes.depth <= self.max_depth[parent.skill_factor], (child.genes.depth, self.max_depth[parent.skill_factor])
        
        
        self.update_parent_profile(child, parent)
        
        return [child]
    
    def update_population_info(self, **kwargs):
        self.num_total_terminals = kwargs['nb_terminals']
        self.tree_creator = FlexTreeFactory(
            num_total_terminals= self.num_total_terminals,
        )
        self.max_depth: int = kwargs['max_depth']
        self.max_length: int = kwargs['max_length']



class PruneMutation(Mutation):
    def __init__(self, atol = 1e-6, rtol= 1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        atol and rtol threshold of std
        '''
        self.atol = atol
        self.rtol = rtol
        
    def __call__(self, parent: Individual):
        child_nodes = []
        
        std = np.sqrt([node.var for node in parent.genes.nodes])
        mean = np.abs([node.mean for node in parent.genes.nodes])
        
        #check atol
        node_idx = std < self.atol
        
        #check rtol
        node_idx = (np.abs((std - mean) / (mean + 1e-12)) < self.rtol) | node_idx
        
        #spare constant aside
        is_constant = np.array([isinstance(node, Constant) or isinstance(node, Percentile) for node in parent.genes.nodes], dtype = bool)
        node_idx = node_idx & (~is_constant)
        
        #spare root aside
        node_idx[-1] = False
        
        #if find no point
        if np.sum(node_idx) < 1:
            return []
        
        
        #convert boolean to idx
        node_idx = np.ravel(np.argwhere(node_idx)).tolist()
        
        #for the sake of simplicity choose the only one node which happend be the deepest node
        point = -1
        max_depth = -1
        for i in node_idx:
            if parent.genes.nodes[i].depth > max_depth and (i < parent.genes.length - 1):
                max_depth = parent.genes.nodes[i].depth
                point = i
                
        
        _, root = parent.genes.split_tree(point)
        
        new_node =  Constant()
        new_node.value = parent.genes.nodes[point].mean 
        new_node.compile()
        child_nodes= root[0] + [new_node] + root[1]
        
        assert len(child_nodes) < parent.genes.length
        child = Individual(Tree(child_nodes, deepcopy= True), task= parent.task, skill_factor= parent.skill_factor)
        self.update_parent_profile(child, parent)
        return [child]
        

class NodeMutation(Mutation):
    def __init__(self, *args, **kwargs):
        ...
    
    def __call__(self, parent: Individual):
        prob = np.ones(parent.genes.length, dtype = np.float64)
        is_leaf = np.array([not node.is_leaf for node in parent.genes.nodes], dtype = np.float64)
        prob = normalize_norm1(prob * is_leaf)
        node = parent.genes.nodes[
            numba_randomchoice_w_prob(prob)
        ]
        assert not node.is_leaf 
        
        funcion_set = FUNCTION_SET if node.is_nonlinear else LINEAR_FUNCTION_SET
        
        candidates = funcion_set[node.arity] if node.arity < 3 else funcion_set[-1]
        
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
        
    def update_population_info(self, **kwargs):
        for mut in self.mutations:
            mut.update_population_info(**kwargs)
    