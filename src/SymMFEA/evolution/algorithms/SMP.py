from .GA import GA
import numpy as np
import numpy as np
from ..ranker import NonDominatedRanker
from ..population import Population
from ..reproducer import SMP_Reproducer, SMPManager
from ..selector import Selector
import matplotlib.pyplot as plt
from .. import offspring_pool
from ...utils.timer import timed
from ...utils import handle_number_of_list
import time
class SMP(GA):
    
    ranker_class = NonDominatedRanker
    reproducer_class = SMP_Reproducer
    selector_class = Selector
    '''
    pass_down_params: attribute to pass down to reproducer
    '''
    pass_down_params: list = ['nb_terminals', "smp", 'p_choose_father', 'num_sub_tasks']
    
    @timed
    def wait(self):
        expected_inqueue = self.expected_generations_inqueue * self.offspring_size * sum(self.nb_inds_each_task)
        with self.multiprocessor.nb_inqueue.get_lock():
            time.sleep(max((self.multiprocessor.nb_inqueue.value - expected_inqueue) / 5000, 0.1))
        
    def generation_step(self, population: Population, generation: int):
        
        if generation > 0:
            #update nb_inds_tasks 
            self.update_nb_inds_tasks(population= population, generation= generation)
            
            #create new individuals
            offsprings = self.reproducer(population)
        
        
            #so far new_born is not necessary because reproducer is not concurrent
            #append to new_born pool
            offspring_pool.new_born.append(offsprings)
            
        else:
            #append to new_born pool
            offspring_pool.new_born.append(population.all())
        
        
        #check if produce new offsprings or finish remaining
        if generation == -1: 
            self.terminated = self.multiprocessor.terminated
            
                
        else:
        
            #submit optimization jobs to multiprocessor
            #optimized inds will be append to optimized pool
            optimize_jobs = offspring_pool.new_born.collect_optimize_jobs()
            self.multiprocessor.submit_jobs(optimize_jobs, async_submit= generation != 0)
        
        
        #first generation need to wait for result
        if generation == 0:
            offspring_pool.optimized.block_until_size_eq(len(optimize_jobs))
        

        #collect optimized offprings
        offsprings, num_offsprings = offspring_pool.optimized.collect_optimized(population.num_sub_tasks)
        
        
        #if there are optimized offsprings
        if num_offsprings:
            
            if generation != 0:
                if generation > 0:
                    #update smp
                    self.reproducer.update_smp(population, offsprings)
                #add offsprings to population
                population.extend(offsprings= offsprings)
                
            
            else:
                #assign optimized individual to population for the first generation
                for subpop, offspring in zip(population, offsprings):
                    subpop.ls_inds = offspring
                    
            
            #collect fitness info for selection
            population.collect_fitness_info()
            
        
            if generation != -1:
                #ranking
                self.ranker(population)
                #select best indivudals
                self.selector(population)
            
            #update info to display
            best_sol = population.collect_best_info
            cur_met = self.examine(best_sol)
            
            
            #update process bar
            self.update_process_bar(population, 
                                    reverse=not self.is_larger_better,
                                    train_steps= self.multiprocessor.train_steps.value,
                                    nb_inqueue= self.multiprocessor.nb_inqueue.value,
                                    processed= self.multiprocessor.processed.value,
                                    multiprocessor_time= self.multiprocessor.times.value,
                                    cur_met = cur_met,
                                    )
            if self.worker_log:
                self.multiprocessor.log()
            
            # #prevent evolve to fast without optimization
            self.wait()
            
            
            
        
        if generation != -1 :    
            #update smp
            self.history_smp.append([self.smp[i].get_smp() for i in range(self.num_sub_tasks)])
            population.update_age()
        
            
    def init_params(self, **kwargs):
        self.history_smp: list = []
        self.p_const_intra: float = kwargs.get('p_const_intra', 0)
        self.min_mutation_rate: float = kwargs.get('min_mutation_rate', 0.1)
        self.delta_lr: int = kwargs.get('delta_lr', 0.1)
        self.num_sub_tasks: int = kwargs['num_sub_task']
        
        self.nb_inds_each_task = np.array(handle_number_of_list(self.nb_inds_each_task, self.num_sub_tasks))
        self.data_sample = handle_number_of_list(self.data_sample, self.num_sub_tasks)
        self.nb_inds_min = np.array(handle_number_of_list(self.nb_inds_min, self.num_sub_tasks))
        
        self.is_larger_better = kwargs['is_larger_better']
        
        # prob choose first parent
        self.p_choose_father = np.full(self.num_sub_tasks, 1 / self.num_sub_tasks) 
        
        
        # Initialize memory smp
        self.smp = [SMPManager(idx_host= i, nb_tasks= self.num_sub_tasks, lr = self.delta_lr, p_const_intra= self.p_const_intra, min_mutation_rate= self.min_mutation_rate) for i in range(self.num_sub_tasks)]

        #save history
        self.history_smp.append([self.smp[i].get_smp() for i in range(self.num_sub_tasks)])

    def display_final_result(self, population:Population):
        super().display_final_result(population)
        self.render_smp()
        self.reproducer.render_p_choose_father()


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

    def handle_params(self, tree_config):        
        for attr in tree_config.keys():
            if not isinstance(tree_config[attr], list): 
                tree_config[attr] = [tree_config[attr] for _ in range(self.num_sub_tasks)]
            
        
        return tree_config