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

    
    #tanh(x0 * x1) + x2    
    nodes_nested = [
        Operand(0), Operand(1), Prod(), Tanh(), Operand(2), Sum(arity= 2)
    ]
    nested_tree = Tree(nodes=nodes_nested, compile = False)
    
    #tanh(x0)    
    nodes_tanh = [
        Operand(0), Tanh()
    ]
    tree_tanh = Tree(nodes=nodes_tanh, compile = False)
    
    #BN(x0)    
    nodes_bn = [
        Operand(0), BatchNorm()
    ]
    nodes_bn[-1].inp_mean = 0.5
    nodes_bn[1].inp_var = 2
    tree_bn = Tree(nodes=nodes_bn, compile = False)
    
    #log(x0)    
    nodes_log = [
        Operand(0), Log()
    ]
    tree_log = Tree(nodes=nodes_log, compile = False)
    
    #relu(x0)    
    nodes_relu = [
        Operand(0), Relu()
    ]
    tree_relu = Tree(nodes=nodes_relu, compile = False)
    
    #x0 * x1    
    nodes_prod = [
        Operand(0), Operand(1), Prod()
    ]
    tree_prod = Tree(nodes=nodes_prod, compile = False)   
    
    #x0 / x1    
    nodes_aq = [
        Operand(0), Operand(1), AQ()
    ]
    tree_aq = Tree(nodes=nodes_aq, compile = False)   
    
    #valve(x0, x1)
    nodes_valve = [
        Operand(0), Operand(1), Valve()
    ]
    tree_valve = Tree(nodes=nodes_valve, compile = False)   
    
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
    generate_input_list((10, 1), size= 10), 'tree_bn'
    ))
    def test_expression_bn(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = (X[:, 0] - 0.5) / (np.sqrt(2) + 1e-12)
        
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
        _log = np.log(margin + 1)
        sign = np.sign(x)
        y = _log * sign
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
        
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 1), size= 10), 'tree_relu'
    ))
    def test_expression_relu(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)
        
        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        
        y = np.maximum(X[:, 0], 0)
        
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
    generate_input_list((10, 2), size= 10), 'tree_aq'
    ))
    def test_expression_aq(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)

        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        y = X[:, 0] / np.sqrt(X[:, 1] ** 2 + 1)
        
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)   
        
    @pytest.mark.parametrize("X, tree", zip_inputs(
    generate_input_list((10, 2), size= 10), 'tree_valve'
    ))
    def test_expression_valve(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        
        print(tree.expression)

        y_expr = tree.callable_expression(X)
        y_normal = tree(X)
        
        x0 = X[:, 0]
        z = np.exp(np.sign(-x0) * x0)
        z = np.where(x0 < 0, z, 1) / (1 + z)
        y = z * X[:, 1]
        
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
        
        y = np.tanh(X[:, 0] * X[:, 1]) + X[:, 2]
        assert is_closed(y_expr, y), (y_expr, y)
        assert is_closed(y_expr, y_normal), (y_expr, y_normal)
    
