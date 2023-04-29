from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
import numpy as np
import pytest
from .utils import *
import random

class TestExpression():
    
    #x0 + x1
    nodes1 = [
        Operand(0), Operand(1), Sum(arity= 2)
    ]
    tree1 = Tree(nodes=nodes1, compile = False)
    
    #tanh(x0 * x1) - x2    
    nodes2 = [
        Operand(0), Operand(1), Prod(), Tanh(), Operand(2), Subtract(arity= 2)
    ]
    tree2 = Tree(nodes=nodes2, compile = False)
    
    
    @pytest.mark.parametrize("X, tree, scale", zip_inputs(
        generate_input_list((10, 2), size= 1), 'tree1', random.random()
        ))
    def test_scale_0(self, X: np.ndarray, tree: Tree, scale:float):
        tree = getattr(self, tree)
        
        
        y_normal = tree(X)
        y = X[:,0] + X[:, 1]
                
        tree.scale(scale)
        
        y_scale = tree(X)
        
        assert is_closed(y_normal, y), (y_normal, y)
        assert is_closed(y_scale, y_normal * scale), (y_scale, y_normal * scale)
        
        
    
    @pytest.mark.parametrize("X, tree, scale", zip_inputs(
    generate_input_list((10, 3), size= 1), 'tree2', random.random()
    ))
    def test_scale_1(self, X: np.ndarray, tree: Tree, scale: float):
        tree = getattr(self, tree)
                
        y_normal = tree(X)
        y = np.tanh(X[:, 0] * X[:, 1]) - X[:, 2]
        
        tree.scale(scale)
        y_scale = tree(X)
        
        assert is_closed(y_normal, y), (y_normal, y)
        assert is_closed(y_scale, y_normal * scale), (y_scale, y_normal * scale)
    
        