from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
import numpy as np
import pytest
from .utils import *


class TestExpression():
    
    #x0 + x1
    nodes_sum = [
        Operand(0), Operand(1), Sum(arity= 2)
    ]
    tree_sum = Tree(nodes=nodes_sum, compile = False)
    
    #x0 - x1
    nodes_subtract = [
        Operand(0), Operand(1), Subtract(arity= 2)
    ]
    tree_sub = Tree(nodes=nodes_subtract, compile = False)
    
    #tanh(x0 * x1) - x2    
    nodes_nested = [
        Operand(0), Operand(1), Prod(), Tanh(), Operand(2), Subtract(arity= 2)
    ]
    nested_tree = Tree(nodes=nodes_nested, compile = False)
    
    #tanh(x0)    
    nodes_tanh = [
        Operand(0), Tanh()
    ]
    tree_tanh = Tree(nodes=nodes_tanh, compile = False)
    
    #log(x0)    
    nodes_log = [
        Operand(0), Log()
    ]
    tree_log = Tree(nodes=nodes_log, compile = False)
    
    #x0 * x1    
    nodes_prod = [
        Operand(0), Operand(1), Prod()
    ]
    tree_prod = Tree(nodes=nodes_prod, compile = False)   
    
    @pytest.mark.parametrize("X, tree", zip_inputs(
        generate_input_list((10, 2), size= 10), 'tree_sum'
        ))
    def test_expression_sum(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = X[:,0] + X[:, 1]
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
        
    @pytest.mark.parametrize("X, tree", zip_inputs(
        generate_input_list((10, 2), size= 10), 'tree_sub'
        ))
    def test_expression_subtract(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = X[:,0] - X[:, 1]
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)        
        
        
        
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 1), size= 10), 'tree_tanh'
    ))
    def test_expression_tanh(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = np.tanh(X[:,0])
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
        
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 1), size= 10), 'tree_log'
    ))
    def test_expression_log(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        
        x = X[:, 0]
        margin = np.abs(x)
        _log = np.where(margin > 1, 1 + np.log(margin), margin)
        sign = np.sign(x)
        y = margin * sign
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
    
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 2), size= 10), 'tree_prod'
    ))
    def test_expression_prod(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)

        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = X[:, 0] * X[:, 1]
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)    

    
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 3), size= 10), 'nested_tree'
    ))
    def test_expression_nested(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        
        y = np.tanh(X[:, 0] * X[:, 1]) - X[:, 2]
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
    
