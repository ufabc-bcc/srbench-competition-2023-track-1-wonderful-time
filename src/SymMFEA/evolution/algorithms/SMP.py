from .GA import GA
import numpy as np
import numpy as np
from ..ranker import SingleObjectiveRanker
from ..population import Population
from ..reproducer import SMP_Reproducer, battle_smp
from ..selector import ElitismSelector
import matplotlib.pyplot as plt
from .. import offsprings_pool

class SMP(GA):
    
    ranker_class = SingleObjectiveRanker
    reproducer_class = SMP_Reproducer
    selector_class = ElitismSelector
    pass_down_params: list = ['nb_terminals', "smp", 'p_choose_father']
        
    def generation_step(self, population: Population, generation: int):
        
        if generation > 0:
            #update nb_inds_tasks 
            self.update_nb_inds_tasks(population= population, generation= generation)
            
            #create new individuals
            offsprings = self.reproducer(population)
            
            #so far new_born is not necessary because reproducer is not concurrent
            
            #append to new_born pool
            offsprings_pool.new_born.append_offsprings(offsprings)
        else:
            #append to new_born pool
            offsprings_pool.new_born.append_offsprings(population.all())
        
        #submit optimization jobs to multiprocessor
        #optimized inds will be append to optimized pool
        optimize_jobs = offsprings_pool.new_born.collect_optimize_job()
        #create callback function to append offsprings when finish optimization
        append_callback = offsprings_pool.optimized.create_append_callback(optimize_jobs=optimize_jobs)
        
        self.multiprocessor.execute(optimize_jobs, append_callback, wait_for_result= generation == 0)
        
        #collect optimized offprings
        offsprings, num_offsprings = offsprings_pool.optimized.collect_optimized(population.num_sub_tasks)
        
        
        #if there are optimized offsprings
        if num_offsprings:
            
            if generation > 0:
                #update smp
                self.reproducer.update_smp(population, offsprings)
                #add offsprings to population
                
                for subpop, offspring in zip(population, offsprings):
                    subpop.ls_inds.extend(offspring)
            
            
            else:
                #assign optimized individual to population for first generation
                for subpop, offspring in zip(population, offsprings):
                    subpop.ls_inds = offspring
            
            
            population.collect_fitness_info()
            
            #ranking
            self.ranker(population)
        
            #select best indivudals
            self.selector(population)
            
            #update info to display
            population.collect_best_info()
            
            #update process bar
            self.update_process_bar(population, reverse=not self.is_larger_better)
        
        #update smp
        self.history_smp.append([self.smp[i].get_smp() for i in range(self.num_sub_tasks)])
            
    def init_params(self, **kwargs):
        self.history_smp: list = []
        self.p_const_intra: float = kwargs['p_const_intra']
        self.delta_lr: int = kwargs['delta_lr']
        self.num_sub_tasks: int = kwargs['num_sub_task']
        self.is_larger_better = kwargs['is_larger_better']
        
        # prob choose first parent
        self.p_choose_father = np.full(self.num_sub_tasks, 1 / self.num_sub_tasks) 
        
        
        # count_eval_stop: nums evals not decrease factorial cost
        # maxcount_es: max of count_eval_stop
        # if count_eval[i] == maxcount_es: p_choose_father[i] == 0
        self.count_eval_stop = [0] * self.num_sub_tasks
        
        
        # Initialize memory smp
        self.smp = [battle_smp(idx_host= i, nb_tasks= self.num_sub_tasks, lr = self.delta_lr, p_const_intra= self.p_const_intra) for i in range(self.num_sub_tasks)]

        #save history
        self.history_smp.append([self.smp[i].get_smp() for i in range(self.num_sub_tasks)])

    def display_final_result(self, population:Population):
        super().display_final_result(population)
        self.render_smp()


    def render_smp(self,  shape = None, title = None, figsize = None, dpi = 100, step = 1, re_fig = False, label_shape= None, label_loc= None):
        
        if title is None:
            title = self.__class__.__name__
        if shape is None:
            shape = (int(np.ceil(self.num_sub_tasks / 3)), 3)
        else:
            assert shape[0] * shape[1] >= self.num_sub_tasks

        if label_shape is None:
            label_shape = (1, self.num_sub_tasks)
        else:
            assert label_shape[0] * label_shape[1] >= self.num_sub_tasks

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

        for idx_task in range(self.num_sub_tasks):
            fig.axes[idx_task].stackplot(
                np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
                [his_smp[
                    np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
                    idx_task, t] for t in range(self.num_sub_tasks + 1)],
                labels = ['Task' + str(i + 1) for i in range(self.num_sub_tasks)] + ["mutation"]
            )
            # plt.legend()
            fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + str(idx_task))
            fig.axes[idx_task].set_xlabel('Generations')
            fig.axes[idx_task].set_ylabel("SMP")
            fig.axes[idx_task].set_ylim(bottom = y_lim[0], top = y_lim[1])


        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(lines, labels, loc = label_loc, ncol = label_shape[1])
        plt.show()
        plt.savefig('SMP.png')
        if re_fig:
            return fig
