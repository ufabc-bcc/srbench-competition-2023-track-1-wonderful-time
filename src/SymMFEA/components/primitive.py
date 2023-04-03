from .functions import Node, FUNCTION_SET, LINEAR_FUNCTION_SET, Operand
import numpy as np
import random

class Primitive:
    def __init__(self, terminal_set = [], num_terminal = 2) -> None:
        self.terminal_set = terminal_set
        self.num_terminal = num_terminal
    
    def get_arity_range(self):
        num_arities = FUNCTION_SET.keys()
        return max(min(num_arities), 1), max(num_arities)
    
    def sample_node(self, a_min: int, a_max: int, get_nonlinear: bool) -> Node:
        
        #prevent nested nonlinear
        if a_max < 2 and get_nonlinear == False: 
            node_cls = Operand
        else:
            f = FUNCTION_SET if get_nonlinear else LINEAR_FUNCTION_SET
            
            candidate_functions = None
            while candidate_functions is None:
                arity = random.randint(a_min, a_max )
                
                candidate_functions = f.get(arity)
        
            node_cls = candidate_functions[random.randint(0, len(candidate_functions) -1)]
        
        if node_cls == Operand:
            return node_cls(index = self.terminal_set[random.randint(0, len(self.terminal_set) - 1)] if len(self.terminal_set) else np.random.randint(0, self.num_terminal))
        return node_cls()