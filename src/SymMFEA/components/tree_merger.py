from .tree import Tree
from ..evolution.population import Individual
from sklearn.linear_model import Lasso
from typing import List
from .data_pool import DataView
from termcolor import colored
from .functions import Sum
import numpy as np
from ..utils.timer import timed
from ..components.metrics import Metric
from pygmo import fast_non_dominated_sorting
from ..utils.functional import normalize_norm1
import warnings

class TreeMerger:
    def __init__(self):
        ...
        
    @timed
    def __call__(self, inds: List[Individual], val_data: DataView, metric: Metric):
        warnings.filterwarnings("ignore")

        y_trains = np.stack([ind(val_data.X_train) for ind in inds]).T
        y_vals = np.stack([ind(val_data.X_val) for ind in inds]).T
        
        #Grid search
        coefs = []
        objs = []
        
        for alpha in [0,1e-12,1e-6,1e-4]:
            
        
            model = Lasso(tol=0, alpha = alpha, fit_intercept=False, positive= True)
            
            model.fit(y_trains, val_data.y_train)
            
            coef_ = model.coef_ 
            
            for p in [1e-3, 1e-2, 1e-1, 2e-1]:
                prune_threshold = max(p, 0.5 / len(inds))
                is_selected = coef_ > prune_threshold
                coef = normalize_norm1(np.where(is_selected, coef_, 0))
                
                coefs.append(coef)

                y_hat = y_vals @ coef
                
                met = metric(val_data.y_val, y_hat)
                
                length = sum([ind.genes.length for i, ind in enumerate(inds) if is_selected[i]])
                objs.append([-met if metric.is_larger_better else met, length])
                
        
        fronts, _, _, _ = fast_non_dominated_sorting(objs)
        
        best_met = 0 
        for idx in fronts[0]: 
            met = - objs[idx][0]
            if met > best_met:
                best_met = met
                best_coef = coefs[idx]
                
            
        
        
        select_idx = best_coef > 0
        
        
        display_str = 'Coef: '
        selected_inds: List[Individual] = []
        selected_weights = []
        
        for ind, c, w in zip(inds, select_idx, best_coef):
            weight = '{:.2f} '.format(w)
            if c:
                selected_inds.append(ind)
                selected_weights.append(w)
                display_str += colored(weight, 'green')
            
            else:
                display_str += colored(weight, 'red')
                
        
        print('=' * 50 + colored('Tree Merger', 'red') +'='*50)
        print(display_str + f'; {str(metric)}:' + colored(' {:.2f}'.format(best_met), 'green'))
        print('=' * 111)
        
        
        #merge into one tree
        
        if len(selected_inds) == 1:
            merged_tree= selected_inds[0].genes
        else:
            nodes = []                                                              
            
            for ind, w in zip(selected_inds, selected_weights):
                ind.scale(w)
                
                nodes.extend(ind.genes.nodes)
            
            #add root
            root = Sum(arity= len(selected_inds))
            root.W = 1
            root.bias = 0
            root.compile()
            nodes.append(root)
            
            merged_tree =  Tree(nodes, deepcopy= True, compile= False)
            
            met = metric(val_data.y_val, merged_tree(val_data.X_val))
            
            assert abs((met - best_met) / (best_met + 1e-12)) < 1e-5, (met, best_met)
            
            print(colored('After merge: {:.2f}, length: {}'.format(met, merged_tree. length), 'red'))
        return merged_tree
            
        
        
        
                
            
        
        
        
        