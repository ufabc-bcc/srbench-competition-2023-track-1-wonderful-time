from .functions import Node, FUNCTION_SET, LINEAR_FUNCTION_SET, Operand, Percentile
import numpy as np
import random
from ..utils.functional import numba_randomchoice_w_prob, normalize_norm1

class Primitive:
    def __init__(self, terminal_set) -> None:
        self.terminal_set = terminal_set
        
    
    
    def sample_node(self, a_min: int, a_max: int, get_nonlinear: bool) -> Node:
        params = {}
        
        #prevent nested nonlinear
        if (a_max < 2 and get_nonlinear == False) or (a_max == 0): 
            r = random.random()
            
            node_cls = Operand if r < 0.5 else Percentile
            # node_cls = Operand
            
            
            
        else:
            f = FUNCTION_SET if get_nonlinear else LINEAR_FUNCTION_SET
            
            candidate_functions = None
            while candidate_functions is None:
                
                if a_max < 2:
                    candidate_functions = f.get(1)    
                else:
                    arity = random.random() 
                    if arity < 1 / 3:
                        candidate_functions = f.get(1)    
                    elif arity < 2 / 3:
                        candidate_functions = f.get(2)
                    else:
                        candidate_functions = f.get(-1)
                        
                        arity = np.array([a_max - a + 1 for a in range(2, a_max + 1)])
                        arity = normalize_norm1(arity ** 3)
                        arity = numba_randomchoice_w_prob(arity) + 2
                        params['arity'] = arity
                        
                        
        
            node_cls = candidate_functions[random.randint(0, len(candidate_functions) -1)]
        
        if node_cls in [Operand, Percentile]:
            return node_cls(index = self.terminal_set[random.randint(0, len(self.terminal_set) - 1)],
                            **params)
        return node_cls(**params)