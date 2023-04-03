import numpy as np
from ..population import Individual
from ...components.tree import Tree
from ...components.functions import Operand, Node, FUNCTION_SET, LINEAR_FUNCTION_SET
from ...utils.functional import numba_randomchoice_w_prob
import random
from typing import List

class Mutation:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, parent: Individual) -> Individual:
        pass
    
    def update_task_info(self, **kwargs):
        pass

class VariableMutation(Mutation):
    def __init__(self, *args, **kwargs):
        self.num_terminal: int 
    
    def __call__(self, parent: Individual):
        child_nodes = []
        for node in parent.genes.nodes:
            if isinstance(node, Operand):
                child_nodes.append(Operand(index = random.randint(0, self.num_terminal - 1)))
            
            else:
                child_nodes.append(Node.deepcopy(node))
        
        return Individual(Tree(child_nodes), task= parent.task)
    
    def update_task_info(self, **kwargs):
        self.num_terminal = kwargs['total_nb_termials']


#NOTE: Numerically unstable
class NodeMutation(Mutation):
    def __init__(self, *args, **kwargs):
        self.num_terminal: int 
    
    def __call__(self, parent: Individual):
        node = parent.genes.nodes[random.randint(0, parent.genes.length - 1)]
        
        funcion_set = FUNCTION_SET if node.is_nonlinear else LINEAR_FUNCTION_SET
        
        candidates = funcion_set[node.arity]
        new_node = Node.deepcopy(node, new_class= candidates[random.randint(0, len(candidates) - 1)])
        
        
        return [Individual(Tree(parent.genes.nodes[:node.id] + [new_node] + parent.genes.nodes[node.id + 1 : ]), task= parent.task, deepcopy = True)]

        
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
    