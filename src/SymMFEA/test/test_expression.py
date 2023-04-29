from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
import numpy as np
import pytest
from .utils import *


class TestExpression():
    
    X1 = np.array ([1, 2, 3])
    
    #x0 + x1
    nodes1 = [
        Operand(0), Operand(1), Sum(arity= 2)
    ]
    tree1 = Tree(nodes=nodes1, compile = False)
    
    
    @pytest.mark.parametrize("X, tree", zip_inputs(
        generate_input_list((10, 2), size= 1), ['tree1' for _ in range(1)]
        ))
    def test_expression_value(self, X: np.ndarray, tree: Tree):
        tree = getattr(self, tree)
        print(tree.expression)
        y_hat = tree.callable_expression(X)
        
        y = X[:,0] + X[:, 1]
        print(y_hat, y)
        assert is_closed(y_hat, y), (y_hat, y)
        
        