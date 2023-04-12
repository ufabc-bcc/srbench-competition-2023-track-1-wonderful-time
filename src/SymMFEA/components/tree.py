from typing import List, Tuple
from .primitive import Primitive
from .functions import Node, Operand, Constant
import numpy as np
from ..utils.timer import *
from ..utils.functional import numba_randomchoice
import math
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
    
    def __call__(self, X: np.ndarray, update_stats= False) -> np.ndarray:
        r'''
        X: matrix with first axis is batch axis
        y: output
        '''
        #init to prevent multiple allocate
        stack = np.empty((self.length, X.shape[0]), dtype = np.float64)
        top = 0
        
        for i, node in enumerate(self.nodes):
            if node.is_leaf:
                stack[top] = node(X, update_stats= update_stats) * self.W[i] + self.bias[i]
                top += 1
            
            else:
                val = node(stack[top - node.arity : top], update_stats= update_stats) * self.W[i] + self.bias[i]
                top -= node.arity
                stack[top] = val
                top += 1
        
        assert top == 1
        return stack[0]
    
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
            self.W = np.ones(len(self.nodes), dtype= np.float64)
            self.bias = np.zeros(len(self.nodes), dtype= np.float64)
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
    def __init__(self, task_idx:int, num_total_terminals: int,  tree_config: dict, *args, **kwargs):
        self.task_idx = task_idx
        for attr in tree_config.keys():
            setattr(self, attr, self.handle_params(tree_config, attr))
            
        self.num_total_terminals: int= num_total_terminals
        
        
        
        if not hasattr(self, 'terminal_set'):
            self.terminal_set: list= [i for i in range(self.num_total_terminals)] 
        else:
            if isinstance(self.terminal_set, float):
                self.terminal_set: list= numba_randomchoice(np.arange(self.num_total_terminals), size= math.ceil(self.terminal_set * self.num_total_terminals), replace= False).tolist()
            else:
                self.terminal_set = self.terminal_set
                
        self.max_depth: int
        self.max_length: int
        
    
    def handle_params(self, tree_config, attr):
        
        #handle terminal set
        
        if type(tree_config[attr]) == list: 
            return tree_config[attr][self.task_idx]
        
        return attr
    
    
    @timed
    def create_tree(self, root_linear_constrant = False):
        pset = Primitive(terminal_set= self.terminal_set, num_total_terminals= self.num_total_terminals)
        
        
        #create root
        a_min = 1
        a_max = self.max_length - 1
        
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
                    cur_a_min = min(1, cur_a_max)
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
    
    def convert_unknown_node(self, node: Node):
        new_node =  Constant()
        new_node.value = node.mean 
        new_node.bias = 0
        
        return new_node
    
    def convert_tree(self, tree: Tree):
        for i, n in enumerate(tree.nodes):
            if isinstance(n, Operand):
                if n.index not in self.terminal_set:
                    tree.nodes[i] = self.convert_unknown_node(n)
        
        #NOTE: redundant compute
        tree.updateNodes()        
        tree.compile()
                

class FlexTreeFactory(TreeFactory):
    def __init__(self, num_total_terminals: int, *args, **kwargs):
        #NOTE: HARD CORD HERE
        self.num_total_terminals = num_total_terminals
        self.terminal_set = [i for i in range(num_total_terminals)]
        
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