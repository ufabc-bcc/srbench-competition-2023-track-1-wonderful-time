from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
import numpy as np
import pytest
from .utils import *
import random
from SymMFEA.components.tree_merger import TreeMerger

class TestExpression():
    
    #x0 + x1
    nodes1 = [
        Operand(0), Operand(1), Sum(arity= 2)
    ]
    tree1 = Tree(nodes=nodes1, compile = False)
    tree1.set_bias(random.random())
    
    #tanh(x0 * x1) + x2    
    nodes2 = [
        Operand(0), Operand(1), Prod(), Tanh(), Operand(2), Sum(arity= 2)
    ]
    tree2 = Tree(nodes=nodes2, compile = False)
    tree2.set_bias(random.random())
    
    tree_merger = TreeMerger()
    
    @pytest.mark.parametrize("X, tree, scale", zip_inputs(
        generate_input_list((10, 2), 1), 'tree1', random.random()
        ))
    def test_scale_0(self, X: np.ndarray, tree: Tree, scale:float):
        tree = getattr(self, tree)
        
        
        y_normal = tree(X)
        y = X[:,0] + X[:, 1] + tree.bias
                
        tree.scale(scale)
        
        y_scale = tree(X)
        
        assert is_closed(y_normal, y), (y_normal, y)
        assert is_closed(y_scale, y_normal * scale), (y_scale, y_normal * scale)
        
        
    
    @pytest.mark.parametrize("X, tree, scale", zip_inputs(
    generate_input_list((10, 3), 1), 'tree2', random.random()
    ))
    def test_scale_1(self, X: np.ndarray, tree: Tree, scale: float):
        tree = getattr(self, tree)
                
        y_normal = tree(X)
        y = np.tanh(X[:, 0] * X[:, 1]) + X[:, 2] + tree.bias
        
        tree.scale(scale)
        y_scale = tree(X)
        
        assert is_closed(y_normal, y), (y_normal, y)
        assert is_closed(y_scale, y_normal * scale), (y_scale, y_normal * scale)
    
    @pytest.mark.parametrize("X, trees, scales", zip_inputs(
    generate_input_list((10, 3), 1), ('tree1', 'tree2'), (1, 1)
    ))
    def test_merge_tree(self, X: np.ndarray, trees: List[Tree], scales: List[float]):
        tree1 = getattr(self, trees[0])
        tree2 = getattr(self, trees[1])
        
        tree1.W[-1] = 1
        tree2.W[-1] = 1
        
        b1, b2 = tree1.bias, tree2.bias
        
        y = tree1(X) * scales[0] + tree2(X) * scales[1]
        print(y[:3], tree1(X)[:3], tree2(X)[:3])
        
        tree = self.tree_merger.merge_trees((tree1, tree2), scales)
        
        print([str(n) for n in tree.nodes])
        print([str(n) for n in tree1.nodes] + [str(n) for n in tree2.nodes])
        print(tree.W)
        print(tree1.W.tolist() + tree2.W.tolist())
        
        y_merged = tree(X)
        assert np.allclose(y, y_merged), (tree.bias, b1, b2, tree.bias - b1 * scales[0] - b2 * scales[1])