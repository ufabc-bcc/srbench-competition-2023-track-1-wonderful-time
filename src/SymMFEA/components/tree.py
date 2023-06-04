from typing import List, Tuple, Callable
from .primitive import Primitive
from .functions import Node, Operand, Constant
import numpy as np
from ..utils.timer import *
from ..utils.progress_bar import SimplificationProgresBar
from ..utils import count_nodes
from ..utils.functional import multiply
from ..components import weight_manager
from sympy import Expr, Float, lambdify
from ..components.column_sampler import ColumnSampler
import os
import numba as nb

nb.njit(nb.types.Tuple((nb.float32, nb.float32))(nb.float32, nb.float32, nb.int64, nb.float32[:]), cache= os.environ.get('DISABLE_NUMBA_CACHE') is None)
def update_node_stats(old_mean, old_var, old_samples, X):	
    new_samples = X.shape[0] + old_samples	
    mean = (old_mean * old_samples + X.sum()) / new_samples	
    var = (old_var * old_samples + X.var() * X.shape[0]) / new_samples	
    return mean, var



class Tree: 
    simplifications = [
        # ('expand', lambda tree: []),
        # ('collect', lambda tree: [{'syms': tree.ls_terminals}]),
    ]
    __slots__ = [
        'nodes', '_W', '_mean', '_var', '_bias',
        'position', 'cached_expression', 'cached_callable_expression',
        'n_samples', 'compiled'
    ]
    def __init__(self, nodes: List[Node], deepcopy = False, init_weight: bool = False, compile:bool = True):
        '''
        compile: copy weight from nodes to weight_manager
        '''
        if deepcopy:
            self.nodes: List[Node] = [Node.deepcopy(n) for n in nodes]
        else:
            self.nodes = nodes
                
        self.updateNodes()
        self.compiled:bool = compile
        if compile:
            self.position = next(weight_manager.WM)
            self.compile(init_weight= init_weight)
        else:
            self._W = np.array([node.value for node in self.nodes], dtype= np.float32)
            self._mean = np.zeros(self.length, dtype = np.float32)
            self._var = np.zeros(self.length, dtype = np.float32)
            self._bias = 0
            
        self.cached_expression: Expr = None
        self.cached_callable_expression: Callable = None
        self.n_samples = 0 

    def __str__(self) -> str:
        s = ''
        for n in self.nodes:
            s += str(n) + ' '
            
        return s
    
    def free_space(self):
        weight_manager.WM.free_space(self.position)
        
    
    def flush(self):
        self.mean[:] = 0
        self.var[:] = 0
        self.n_samples = 0
    
    @property
    def num_nonlinear(self):
        num = 0
        for node in self.nodes:
            if node.is_nonlinear:
                num+=1
        return num
    
    @property
    def bias(self):
        if self.compiled:
            return weight_manager.WM.tree_bias[self.position]
        else:
            return self._bias
        
    def set_bias(self, val):
        if self.compiled:
            weight_manager.WM.tree_bias[self.position] = val
        else:
            self._bias = val
        
    @property
    def W(self):
        if self.compiled:
            return weight_manager.WM.weight[self.position][:len(self.nodes)]
        else:
            return self._W

    @property
    def mean(self):
        if self.compiled:
            return weight_manager.WM.mean[self.position][:len(self.nodes)]
        else:
            return self._mean
        
    @property
    def var(self):
        if self.compiled:
            return weight_manager.WM.var[self.position][:len(self.nodes)]
        else:
            return self._var
    
    @property
    def dW(self):
        assert self.compiled
        return weight_manager.WM.dW[self.position][:len(self.nodes)]

    @property
    def momentum(self):
        assert self.compiled
        return weight_manager.WM.momentum[self.position][:len(self.nodes)]    

    @property
    def velocity(self):
        assert self.compiled
        return weight_manager.WM.velocity[self.position][:len(self.nodes)]

    @property
    def length(self):
        return self.nodes[-1].length + 1
    
    @property
    def depth(self):
        return self.nodes[-1].depth
    
    def __call__(self, X: np.ndarray, update_stats= False, training= False, check_stats= False) -> np.ndarray:
        r'''
        X: matrix with first axis is batch axis
        y: output
        '''
        #init to prevent multiple allocate
        stack = np.empty((self.length, X.shape[0]), dtype = np.float32)
        top = 0
        
        W = self.W 
        mean = self.mean 
        var = self.var
        
        for i, node in enumerate(self.nodes):
            if node.is_leaf:
                stack[top] = node(X, training= training) * W[i] 
                
            
            else:
                val = multiply(node(stack[top - node.arity : top], training= training), W[i])
                top -= node.arity
                stack[top] = val
                
            if update_stats:
                self.mean[i], self.var[i] = update_node_stats(self.mean[i], self.var[i], self.n_samples, stack[top])
                
            #only run when test
            if check_stats:
                if not isinstance(node, Constant):
                    raw = stack[top]
                    mean = np.mean(raw)
                    var = np.var(raw)
                    assert abs(mean - self.mean[i]) / (abs(self.mean[i]) + 1e-12) < 1e-3
                    assert abs(var - self.var[i]) / (abs(self.var[i]) + 1e-12) < 1e-3
                    
            top += 1
        
        if update_stats:
            self.n_samples += X.shape[0]
        
        assert top == 1
        return stack[0] + self.bias
    

    @property
    def expression(self) -> Expr:
        if self.cached_expression is not None:
            return self.cached_expression
        
        print('\n' + ('=' * 100))
        
        
        with SimplificationProgresBar(simplification_list= self.simplifications) as (progress_bar, progress):
            stack: List[Expr] = [None] * self.length
            top = 0
            W = self.W 
            
            for i, node in enumerate(self.nodes):
                if node.is_leaf:
                    stack[top] = node.expression() * Float(W[i]) 
                    top += 1
                
                else:
                    val = node.expression(stack[top - node.arity : top]) * Float(W[i]) 
                    top -= node.arity
                    stack[top] = val
                    top += 1
            
            assert top == 1
            expr = stack[0]
            first_length = count_nodes(expr)
            
            progress_bar.update(count_nodes(expr))
            for (simplify, get_args_funcs) in progress:
                args = get_args_funcs(self)
                for kwargs in args:
                    
                    progress_bar.update_what_iam_doing(simplify)
                    attempt: Expr = getattr(expr, simplify)(**kwargs)
                    
                    if count_nodes(attempt) < count_nodes(expr):
                        expr = attempt
                    
                    progress_bar.update(count_nodes(expr))
                
                if simplify == self.simplifications[-1][0]:
                    progress_bar.set_finished()
        
        print('Number of nodes: Before: ' + colored('{:,}'.format(first_length), 'red') + ' After: ' + colored('{:,}'.format(count_nodes(expr)), 'green'))
        print('=' * 100)
        self.cached_expression = expr
        return expr
    
    def callable_expression(self, nvars:int = None) -> Callable:
        if self.cached_callable_expression is not None and (nvars is None):
            return self.cached_callable_expression
        
        vars = [f"x{i}" for i in range(self.largest_terminal if nvars is None else nvars)]
        
        f = lambdify(vars, self.expression, 'numpy')
        
        def infer(X):
            nonlocal f 
            return f(*[X[:, i] for i in range(X.shape[1])])
        return infer
    
    # @property
    # def attrs(self):
    #     return [node.attrs for node in self.nodes]  
    
    @property
    def largest_terminal(self) -> int:
        rs = 0
        for node in self.nodes:
            if isinstance(node, Operand):
                rs = max(rs, node.index)

        return rs + 1
    
    @property
    def ls_terminals(self) -> list:
        return list(set([node.index for node in self.nodes if isinstance(node, Operand)]))
    
    def updateNodes(self):
        for i, node in enumerate(self.nodes):
            node.depth = 1
            node.length = node.arity
            node.parent = 0
            node.id = i
            if node.is_leaf:
                continue
            
            j = i - 1
            
            for _ in range(node.arity):
                child = self.nodes[j]
                node.length += child.length
                node.depth = max (node.depth, child.depth)
                child.parent = i 
                j-= child.length + 1
                
            node.depth += 1
        
        assert self.length == len(self.nodes)
        
    
    def get_branch_nodes(self, i, return_both = False):
        return self.nodes[i - self.nodes[i].length : i + 1]
        
    
    def split_tree(self, point) -> Tuple[List[Node], Tuple[List[Node], List[Node]]]:
        branch = self.nodes[point - self.nodes[point].length : point + 1]
        root = self.nodes[: point - self.nodes[point].length], self.nodes[ point + 1 : ]
        return branch, root
        
    def compile(self, init_weight = False):
        r'''
        compile from list of node manner to vector manner
        after compile value and bias from node doesn't matter
        '''
        W = self.W
        
        if init_weight:
            W[:] = np.ones(len(self.nodes), dtype= np.float32)
        else:
            for i, node in enumerate(self.nodes):
                W[i] = node.value
        
        for i, node in enumerate(self.nodes):
            node.compile(self)
        
        self.update_best_tree()
        
        
    def scale(self, scale_factor:float):
        self.W[self.length - 1] = self.W[self.length - 1] * scale_factor
        self.set_bias(self.bias * scale_factor)

            
    def run_check_expression(self, data):
        if os.environ.get('DEBUG'):
            X = data.X_val
                        
            assert np.allclose(self(X), self.callable_expression(nvars = X.shape[1])(X))
      
    def update_best_tree(self):
        
        weight_manager.WM.best_weight[self.position] = weight_manager.WM.weight[self.position]
        weight_manager.WM.best_tree_bias[self.position] = weight_manager.WM.tree_bias[self.position]
    
    def rollback_best(self):
        weight_manager.WM.weight[self.position] = weight_manager.WM.best_weight[self.position]
        weight_manager.WM.tree_bias[self.position] = weight_manager.WM.best_tree_bias[self.position]
    

class TreeFactory:
    def __init__(self, task_idx:int, num_total_terminals: float, tree_config: dict, column_sampler: ColumnSampler, *args, **kwargs):
        self.task_idx = task_idx
        for attr in tree_config.keys():
            setattr(self, attr, tree_config[attr][self.task_idx])
            
        self.num_total_terminals: int= num_total_terminals
        
        self.terminal_set: list= column_sampler.sample(size= self.num_columns)

                
        self.max_depth: int
        self.max_length: int
        
    

    
    
    @timed
    def create_tree(self, root_linear_constrant = False):
        pset = Primitive(terminal_set= self.terminal_set)
        
        
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
        
        return new_node
    
    def convert_tree(self, tree: Tree):
        for i, n in enumerate(tree.nodes):
            if isinstance(n, Operand):
                if n.index not in self.terminal_set:
                    tree.nodes[i] = self.convert_unknown_node(tree.nodes[i])
        
        #NOTE: redundant compute
        tree.updateNodes()        
        tree.compile()
                

class FlexTreeFactory(TreeFactory):
    def __init__(self, num_total_terminals: int, *args, **kwargs):
        self.num_total_terminals = num_total_terminals
        self.terminal_set: List[int]
        
        
    def update_config(self, max_depth: int, max_length: int, terminal_set: List[int]):
        self.terminal_set = terminal_set
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