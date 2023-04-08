from typing import List, Tuple
from .primitive import Primitive
from .functions import Node
import numpy as np


    
def create_mask_from_index(idx: np.ndarray, length: int):
    assert np.max(idx) < length, (idx, length)
    mask = np.zeros(length, dtype= np.float64)
    mask[idx] = 1
    return mask

class Tree: 
    def __init__(self, nodes: List[Node], deepcopy = False, mask: np.ndarray = None, init_weight: bool = False) -> None:
        if deepcopy:
            self.nodes: List[Node] = [Node.deepcopy(n) for n in nodes]
        else:
            self.nodes = nodes
        self.updateNodes()
        self.compile(mask= mask, init_weight= init_weight)
        self.W: np.ndarray
        self.bias: np.ndarray
        self.dW: np.ndarray
        self.dB: np.ndarray
        self.isPrime: bool = False
    
    def __str__(self) -> str:
        s = ''
        for n in self.nodes:
            s += str(n) + ' '
            
        return s
    
    
    @property
    def length(self):
        return self.nodes[-1].length + 1
    
    @property
    def depth(self):
        return self.nodes[-1].depth
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        r'''
        X: matrix with first axis is batch axis
        y: output
        '''
        stack = []
        
        for i, node in enumerate(self.nodes):
            if node.is_leaf:
                stack.append(node(X))
            
            else:
                val = node(stack[-node.arity : ]) * self.W[i] + self.bias[i]
                stack = stack[:-node.arity]
                stack.append(val)
        
        return stack[-1]
    
    def updateNodes(self):
        for i, node in enumerate(self.nodes):
            node.depth = 1
            node.length = node.arity
            node.parent = 0
            node.id = i
            if node.is_leaf:
                continue
            
            j = i - 1
            
            for k in range(node.arity):
                child = self.nodes[j]
                node.length += child.length
                node.depth = max (node.depth, child.depth)
                child.parent = i 
                j-= child.length + 1
                
            node.depth += 1
        
        assert self.length == len(self.nodes)
        
    def remove_mask(self):
        self.node_grad_mask = np.ones(self.length, dtype= np.float64)
    
    def get_branch_nodes(self, i, return_both = False):
        return self.nodes[i - self.nodes[i].length : i + 1]
        
    
    def split_tree(self, point) -> Tuple[List[Node], Tuple[List[Node], List[Node]]]:
        branch = self.nodes[point - self.nodes[point].length : point + 1]
        root = self.nodes[: point - self.nodes[point].length], self.nodes[ point + 1 : ]
        return branch, root
        
    def compile(self, mask:np.ndarray = None, init_weight = False):
        r'''
        compile from list of node manner to vector manner
        mask: index to have grad, other will turn off grad
        after compile value and bias from node doesn't matter
        '''
        self.W = np.empty(self.length, dtype = np.float64)
        self.bias = np.empty(self.length, dtype = np.float64)
        self.dW = np.empty(self.length, dtype = np.float64)
        self.dB = np.empty(self.length, dtype = np.float64)
        self.node_grad_mask = np.ones(self.length, dtype = np.float64) if mask is None \
                        else create_mask_from_index(mask, self.length)
        
        if init_weight:
            self.W = np.random.normal(0, 1, len(self.nodes))
            self.bias = np.random.normal(0, 1, len(self.nodes))
        else:
            for i, node in enumerate(self.nodes):
                self.W[i] = node.value
                self.bias[i] = node.bias
        
        for i, node in enumerate(self.nodes):
            node.compile(self)
        
        self.bestW = self.W 
        self.bestBias = self.bias
      
    def update(self):
        self.bestW = self.W 
        self.bestBias = self.bias   
    
    def rollback_best(self):
        self.isPrime = True
        self.W = self.bestW 
        self.bias = self.bestBias

class TreeFactory:
    def __init__(self, task_idx:int,  tree_config: dict, *args, **kwargs):
        self.task_idx = task_idx
        self.terminal_set: list
        self.num_total_terminals: int
        self.max_depth: int
        self.max_length: int
        for attr in tree_config.keys():
            setattr(self, attr, self.handle_params(tree_config, attr))
    
    def handle_params(self, tree_config, attr):
        
        #handle terminal set
        
        if type(tree_config[attr]) == list: 
            return tree_config[attr][self.task_idx]
        
        return attr
    
    
    def create_tree(self, root_linear_constrant = False):
        pset = Primitive(terminal_set= self.terminal_set, num_total_terminals= self.num_total_terminals)
        a_min, a_max = pset.get_arity_range()
        
        #create root
        a_min = min(a_min, self.max_length - 1)
        a_max = min(a_max, self.max_length - 1)
        root = pset.sample_node(a_min, a_max, get_nonlinear= not root_linear_constrant)
        
        num_open_nodes = root.arity
        
        
        #list of tuple node, depth, firstChildIndex
        ls_nodes = [[root, 0, 1]]
        
        for i in range(self.max_length):
            if num_open_nodes:
                node, depth, _ = ls_nodes[i]
                ls_nodes[i][2] = len(ls_nodes)
                d = depth + 1
                for _ in range(node.arity):
                    cur_a_max = min(a_max, self.max_length - num_open_nodes - i - 1) if d < (self.max_depth - 1) else 0
                    cur_a_min = min(a_min, cur_a_max)
                    child_node = pset.sample_node(cur_a_min, cur_a_max, get_nonlinear= not node.is_nonlinear)
                    ls_nodes.append([child_node, d, None])
                    num_open_nodes += child_node.arity
                
                num_open_nodes -= 1
        
        #transform into postfix
        postfix = [None] * len(ls_nodes)
        
        idx = len(ls_nodes)
        def fill_postfix(indexdx):
            nonlocal  idx
            node, _, firstChildIndex = ls_nodes[indexdx]
            idx -= 1 
            postfix[idx] = node
            if node.is_leaf:
                return
            
            for i in range(firstChildIndex, firstChildIndex + node.arity):
                fill_postfix(i)
        
        fill_postfix(0)
        
        return Tree(nodes = postfix, init_weight = True)

class FlexTreeFactory(TreeFactory):
    def update_config(self, max_depth: int, max_length: int):
        self.max_depth = max_depth
        self.max_length = max_length
        

def get_possible_range(tree:Tree, point: int, max_length: int, max_depth: int):
    tar_depth = tree.genes.nodes[point].depth
    tar_level = tree.genes.depth - tar_depth
    max_depth = max_depth - tar_level
    
    tar_length = tree.genes.nodes[point].length 
    tar_remain_length = tree.genes.length - tar_length
    max_length = max_length - tar_remain_length
    return max_length, max_depth