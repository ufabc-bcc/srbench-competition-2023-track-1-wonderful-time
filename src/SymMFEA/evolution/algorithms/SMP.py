from .GA import GA
import numpy as np
import numpy as np
from ..ranker import SingleObjectiveRanker
from ..population import Population
from ..reproducer import Reproducer
from ..task import Task
from ..selector import ElitismSelector
from ...components.trainer import Loss, GradOpimizer
from ...components.metrics import Metric
import matplotlib.pyplot as plt
from typing import List

class battle_smp:
    def __init__(self, idx_host: int, nb_tasks: int, lr, p_const_intra) -> None:
        assert idx_host < nb_tasks
        self.idx_host = idx_host
        self.nb_tasks = nb_tasks

        #value const for intra
        self.p_const_intra = p_const_intra
        self.lower_p = 0.1/(self.nb_tasks + 1)

        # smp without const_val of host
        self.sum_not_host = 1 - 0.1 - p_const_intra
        self.SMP_not_host: np.ndarray = ((np.zeros((nb_tasks + 1, )) + self.sum_not_host)/(nb_tasks + 1))
        self.SMP_not_host[self.idx_host] += self.sum_not_host - np.sum(self.SMP_not_host)

        smp_return : np.ndarray = np.copy(self.SMP_not_host)
        smp_return[self.idx_host] += self.p_const_intra
        smp_return += self.lower_p

        self.SMP_include_host = smp_return
        self.lr = lr

    def get_smp(self) -> np.ndarray:
        return np.copy(self.SMP_include_host)
    
    def update_SMP(self, Delta_task, count_Delta_tasks):
        '''
        Delta_task > 0 
        '''

        if np.sum(Delta_task) != 0:         
            # newSMP = np.array(Delta_task) / (self.SMP_include_host)
            newSMP = (np.array(Delta_task) / (np.array(count_Delta_tasks) + 1e-50))
            newSMP = newSMP / (np.sum(newSMP) / self.sum_not_host + 1e-50)

            self.SMP_not_host = self.SMP_not_host * (1 - self.lr) + newSMP * self.lr
            
            self.SMP_not_host[self.idx_host] += self.sum_not_host - np.sum(self.SMP_not_host)
            
            smp_return : np.ndarray = np.copy(self.SMP_not_host)
            smp_return[self.idx_host] += self.p_const_intra
            smp_return += self.lower_p
            self.SMP_include_host = smp_return

        return self.SMP_include_host


def render_smp(self,  shape = None, title = None, figsize = None, dpi = 100, step = 1, re_fig = False, label_shape= None, label_loc= None):
    
    if title is None:
        title = self.__class__.__name__
    if shape is None:
        shape = (int(np.ceil(len(self.tasks) / 3)), 3)
    else:
        assert shape[0] * shape[1] >= len(self.tasks)

    if label_shape is None:
        label_shape = (1, len(self.tasks))
    else:
        assert label_shape[0] * label_shape[1] >= len(self.tasks)

    if label_loc is None:
        label_loc = 'lower center'

    if figsize is None:
        figsize = (shape[1]* 6, shape[0] * 5)

    fig = plt.figure(figsize= figsize, dpi = dpi)
    fig.suptitle(title, size = 15)
    fig.set_facecolor("white")
    fig.subplots(shape[0], shape[1])

    his_smp:np.ndarray = np.copy(self.history_smp)
    y_lim = (-0.1, 1.1)

    for idx_task, task in enumerate(self.tasks):
        fig.axes[idx_task].stackplot(
            np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
            [his_smp[
                np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
                idx_task, t] for t in range(len(self.tasks) + 1)],
            labels = ['Task' + str(i + 1) for i in range(len(self.tasks))] + ["mutation"]
        )
        # plt.legend()
        fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + task.name)
        fig.axes[idx_task].set_xlabel('Generations')
        fig.axes[idx_task].set_ylabel("SMP")
        fig.axes[idx_task].set_ylim(bottom = y_lim[0], top = y_lim[1])


    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(lines, labels, loc = label_loc, ncol = label_shape[1])
    plt.show()
    if re_fig:
        return fig


class SMP(GA):
    
    def generation_step(self, population: Population):
        #create new individuals
        self.reproducer(population)
        
        #train and get fitness from individuals
        population.optimize()
        population.collect_fitness_info()
        
        
        #ranking
        self.ranker(population)
    
        #select best indivudals
        self.selector(population)
        
        #update info to display
        population.collect_best_info()
        
    def init_params(self, population: Population, **kwargs):
        self.history_smp: list = []
        self.p_const_intra = kwargs['p_const_intra']
        self.delta_lr = kwargs['delta_lr']

        
        # prob choose first parent
        self.p_choose_father = np.ones((len(self.tasks), ))/ len(self.tasks)
        
        
        # count_eval_stop: nums evals not decrease factorial cost
        # maxcount_es: max of count_eval_stop
        # if count_eval[i] == maxcount_es: p_choose_father[i] == 0
        self.count_eval_stop = [0] * len(self.tasks)
        
        
        # Initialize memory M_smp
        M_smp = [self.battle_smp(i, len(self.tasks), self.delta_lr, self.p_const_intra) for i in range(len(self.tasks))]

        #save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])

        # Delta epoch
        self.Delta:List[List[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()
        self.count_Delta: List[List[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()
