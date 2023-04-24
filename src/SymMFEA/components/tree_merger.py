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

class TreeMerger:
    def __init__(self):
        ...
        
    @timed
    def __call__(self, inds: List[Individual], val_data: DataView, metric: Metric):
        y_trains = np.stack([ind(val_data.X_train) for ind in inds]).T
        y_vals = np.stack([ind(val_data.X_val) for ind in inds]).T
        
        #Grid search
        coefs = []
        objs = []
        
        for alpha in [1,2,3,4]:
            for prune_threshold in [1e-2, 1e-1, 2e-1, 5e-1]:
        
                model = Lasso(alpha = alpha, positive= True)
                
                model.fit(y_trains, val_data.y_train)
                
                coef = model.coef_ 
                
                is_selected = coef > prune_threshold
                coef = np.where(is_selected, coef, 0)
                
                coefs.append(coef)

                y_hat = y_vals @ coef
                
                met = metric(val_data.y_val, y_hat)
                
                length = sum([ind.genes.length for i, ind in enumerate(inds) if is_selected[i]])
                objs.append([-met if metric.is_larger_better else met, length])
                
        
        fronts, _, _, _ = fast_non_dominated_sorting(objs)
        
        best_met = 0 
        for idx in fronts[0]: 
            met = objs[idx][0]
            if met > best_met:
                best_met = met
                best_coef = coefs[idx]
                
            
        
        
        select_idx = best_coef > 0
        
        
        display_str = 'Coef: '
        selected_inds: List[Individual] = []
        selected_weights = []
        
        for ind, c, w in zip(inds, select_idx, coef):
            weight = '{:.2f} '.format(w)
            if c:
                selected_inds.append(ind)
                selected_weights.append(w)
                display_str += colored(weight, 'green')
            
            else:
                display_str += colored(weight, 'red')
                
        
        
        print(display_str)
        
        
        
        #merge into one tree
        
        nodes = []
        
        for ind, w in zip(selected_inds, selected_weights):
            ind.scale(w)
            
            nodes.extend(ind.genes.nodes)
        
        #add root
        root = Sum(arity= len(selected_inds))
        root.W = 1
        root.bias = 0
        nodes.append(root)
        
        merged_tree =  Tree(nodes, deepcopy= True)
        
        assert abs(abs(metric(val_data.y_val, Tree(val_data.X_val))) - abs(best_met)) < 1e-5
        
        print('After merge: {:.2f}, length: {}'.format(str(metric), merged_tree. length))
        return merged_tree
            
        
        
        
                
            
        
        
        
        