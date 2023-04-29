from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
import numpy as np
import pytest
from .utils import *


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
    
    
    @pytest.mark.parametrize("X, tree", zip_inputs(
        generate_input_list((10, 2), size= 1), 'tree1'
        ))
    def test_expression_value_0(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = X[:,0] + X[:, 1]
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
        
    
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 3), size= 1), 'tree2'
    ))
    def test_expression_value_1(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        
        y = np.tanh(X[:, 0] * X[:, 1]) - X[:, 2]
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
        