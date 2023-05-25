import numpy as np
from ..utils.timer import timed
from ..utils.functional import normalize_norm1, numba_randomchoice_w_prob, ONE
import random
import math
class ColumnSampler:
    
    
    @timed
    def fit(self, X: np.ndarray):
        self.corrcoef = np.corrcoef(X.T)
        self.dim = X.shape[1]
        
        self.coposs = np.empty_like(self.corrcoef)
        for i in range(self.coposs.shape[0]):
            #lower chance for high correlated variables
            self.coposs[i] = normalize_norm1(1 - np.abs(self.corrcoef[i]))
        
    
    
    @timed
    def sample(self, size: float):
        assert size > 0 and (size <= 1), size
        nb_collect = max(math.ceil(size * self.dim), 1)
        terminals = [random.randint(0, self.dim -1)]
        
        while len(terminals) < nb_collect:
            #remove selected index
            poss = np.delete(self.coposs[terminals], terminals, axis = 1)
            corres_idx = np.delete(np.arange(self.dim), terminals, axis = 0)
            
            poss = normalize_norm1(np.mean(poss, axis = 0) )
            
            
            t = corres_idx[numba_randomchoice_w_prob(poss)]

            assert t not in terminals
            
            terminals.append(t)
        
        return terminals
            
            
            
                    
        