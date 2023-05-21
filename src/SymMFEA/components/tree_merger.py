from .tree import Tree
from ..evolution.population import Individual
from sklearn.linear_model import Lasso
from typing import List, Iterable
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
        
        alphas = [0] + [4 ** (-i) for i in range(0, 20)]
        thresholds = [0.1 * i for i in range(1, 10)]
        for alpha in alphas:
            
        
            model = Lasso(tol=0, alpha = alpha, fit_intercept=False, positive= True)
            
            model.fit(y_trains, val_data.y_train)
            
            coef_ = model.coef_ 
            
            for p in thresholds:
                prune_threshold = max(p, 0.5 / len(inds))
                is_selected = coef_ > prune_threshold
                if sum(is_selected) == 0:
                    continue
                coef = normalize_norm1(np.where(is_selected, coef_, 0))
                
                coefs.append(coef)

                y_hat = y_vals @ coef
                
                met = metric(val_data.y_val, y_hat)
                met = -met if metric.is_larger_better else met
                
                length = sum([ind.genes.length for i, ind in enumerate(inds) if is_selected[i]])
                objs.append([met, -length])
                
        for i in range(len(inds)):
            coef = np.zeros(len(inds), dtype= np.float64)
            coef[i] = 1
            
            coefs.append(coef)
            
            objs.append([inds[i].main_objective, -inds[i].genes.length])
        
        
        fronts, _, _, _ = fast_non_dominated_sorting(-np.array(objs))
        
        best_met = -10000000000
        best_length = 1000000000
        for idx in fronts[0]: 
            #get objectives
            met, length = objs[idx][0], -objs[idx][1]
            
            if met > best_met:
                if met - best_met > metric.better_tol or length < best_length:
                    best_met = met
                    best_coef = coefs[idx]
                    best_length = length
                
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
        
        
        
        #merge into one tree
        
        if len(selected_inds) == 1:
            merged_tree= selected_inds[0].genes
            met = metric(val_data.y_val, merged_tree(val_data.X_val))
        else:
            merged_tree = self.merge_trees([ind.genes for ind in selected_inds], selected_weights)
            
            met = metric(val_data.y_val, merged_tree(val_data.X_val))
            
        met = -met if metric.is_larger_better else met
        
        assert abs((met - best_met) / (best_met + 1e-12)) < 1e-5, (met, best_met)
            
        print(colored('After merge: {:.2f}, length: {}'.format(met, merged_tree.length), 'green'))
            
        print('=' * 111)
        return merged_tree
    
    def merge_trees(self, trees: Iterable[Tree], weights: Iterable[float]):
        assert len(trees) == len(weights)
        nodes = []                                                              
        biases = []
        for tree, w in zip(trees, weights):
            tree.scale(w)
            nodes.extend(tree.nodes)
            biases.append(tree.bias)
        
        #add root
        root = Sum(arity= len(trees))
        root.value = 1
        root.compile()
        nodes.append(root)
        
        merged_tree =  Tree(nodes, deepcopy= True, compile= False)
        merged_tree.set_bias(np.sum(biases))
        return merged_tree

        
        
        
        
                
            
        
        
        
        