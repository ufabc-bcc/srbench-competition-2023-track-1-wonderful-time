import numpy as np
from ..ranker import SingleObjectiveRanker
from ..population import Population, Individual
from ..reproducer import Reproducer
from ..task import Task
from ..selector import ElitismSelector
from ...components.trainer import Loss, GradOpimizer
from ...components.metrics import Metric
from ...utils import GAProgressBar, draw_tree
import matplotlib.pyplot as plt
class GA:
    def __init__(self, seed:float = None,
                 reproducer_config:dict = {},
                 ranker_config:dict = {},
                 selector_config: dict = {}):
        '''
        reproducer_config = {
            crossover,
            mutation,
            crossover_size,
            mutation_size,
        }
        '''
        # initial history of factorial cost
        self.seed = seed

        # Add list abstract 
        
        self.generations = 100 # represent for 100% 
        self.ranker = SingleObjectiveRanker(**ranker_config)
        self.reproducer = Reproducer(**reproducer_config)
        self.selector = ElitismSelector(**selector_config)
        self.nb_tasks = 1
        self.final_solution = None
    
    
    def update_process_bar(self, population: Population, reverse:bool):
        self.pbar.update([subPop.max_main_objective for subPop in population], reverse= reverse)
    

    def display_final_result(self, population: Population):
        best_trees, _ = population.get_best_trees()
        sqrt = np.sqrt(len(best_trees)).item()
        nb_columns = int(sqrt)
    
        nb_rows = nb_columns if sqrt == nb_columns else nb_columns + 1
        fig, axs = plt.subplots(nb_rows, nb_columns, figsize = (20 * nb_rows, 10 * nb_columns), squeeze = False)
        for i in range(nb_rows):
            for j in range(nb_columns):
                draw_tree(best_trees[i * nb_columns + j].genes, axs[i * nb_columns, j])
        
        plt.show()
        
    def update_nb_inds_tasks(self, population:Population, generation: int):
        population.update_nb_inds_tasks([int(
                int(min((self.nb_inds_min - self.nb_inds_each_task)/(self.nb_generations - 1)* (generation - 1) + self.nb_inds_each_task, self.nb_inds_each_task))
            )] * self.num_sub_tasks)
    
    def generation_step(self, population: Population, generation: int):
        
        #update nb_inds_tasks 
        self.update_nb_inds_tasks(population= population, generation= generation)
        
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
        
    def init_params(self, population:Population, **kwargs):
        self.num_sub_tasks: int = 1
    
    def finetune(self, ind:Individual):
        ind.finetune()
        
            
    def fit(self, X: np.ndarray,
            y: np.ndarray,
            loss: Loss,
            optimzier: GradOpimizer,
            metric: Metric,
            batch_size:int = 10,
            shuffle:bool = True,
            nb_not_improve:int = None,
            steps_per_gen: int = 10,
            nb_generations: int = 100,
            nb_inds_each_task: int = 100,
            nb_inds_min: int = 10,
            tree_config: dict = {},
            visualize:bool = False,
            test_size: float = 0.2, 
            finetune_steps: int = 5000,
            finetune_decay_lr: float = 100,
            **params,
            ):
        '''
        X: 2d array
        y: 1d array
        tree: {
            max_length: \n
            max_depth
        }
        steps_per_gen: backprop step per generation
        '''
        
        self.nb_inds_min = nb_inds_min
        self.nb_generations = nb_generations
        self.nb_inds_each_task = nb_inds_each_task
        
                
        # initialize population
        population = Population(
            nb_inds_tasks = nb_inds_each_task, 
            task = Task(X, y, loss, optimzier, metric, steps_per_gen= steps_per_gen,
                        batch_size= batch_size, test_size= test_size,
                        shuffle= shuffle, nb_not_improve= nb_not_improve),
            tree_config= tree_config,
        )
        
        self.reproducer.update_task_info(
            **tree_config,
            total_nb_termials=X.shape[1]
        )
        
        self.init_params(**params, population = population)
        
        self.pbar = GAProgressBar(num_iters = nb_generations, metric_name = str(metric))
        
        try:
            for generation in self.pbar.pbar:
                self.generation_step(population, generation)
                self.update_process_bar(population, reverse = not metric.is_larger_better)
            
            best_trees, self.final_solution = population.get_best_trees()
            
            #finetune solution
            self.final_solution.finetune(finetune_steps= finetune_steps, decay_lr= finetune_decay_lr, verbose = True)
        
        except KeyboardInterrupt:
            best_trees, self.final_solution = population.get_best_trees()
            
        else:
            if visualize:
                self.display_final_result(population)
        
    def predict(self, X: np.ndarray):
        return self.final_solution(X)