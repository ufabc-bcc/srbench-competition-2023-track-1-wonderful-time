import numpy as np
from ..ranker import SingleObjectiveRanker, Ranker
from ..population import Population, Individual
from ..reproducer import Reproducer
from ..task import Task
from ..selector import ElitismSelector, Selector
from ...components.trainer import Loss, GradOpimizer
from ...components.metrics import Metric
from ...utils import GAProgressBar, draw_tree
import matplotlib.pyplot as plt
from ...utils.timer import *
from ...components.weight_manager import initWM
from termcolor import colored
from ...components.multiprocessor import Multiprocessor
from ..offsprings_pool import initOffspringsPool
from ..import offsprings_pool
import traceback

#NOTE: probably generation step outdated and GA is not working
class GA:
    ranker_class = SingleObjectiveRanker
    reproducer_class = Reproducer
    selector_class = ElitismSelector
    pass_down_params: list = ['nb_terminals']

    def __init__(self, seed: float = None,
                 reproducer_config: dict = {},
                 ranker_config: dict = {},
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
        self.ranker: Ranker = self.ranker_class(**ranker_config)
        self.reproducer: Reproducer = self.reproducer_class(
            **reproducer_config)
        self.selector: Selector = self.selector_class(**selector_config)
        self.final_solution = None
        self.terminated: bool = False

    def update_process_bar(self, population: Population, reverse: bool, **kwargs):
        self.pbar.update(
            [subPop.max_main_objective for subPop in population], reverse=reverse, **kwargs)

    def display_final_result(self, population: Population):
        best_trees, _ = population.get_best_trees()
        sqrt = np.sqrt(len(best_trees)).item()
        nb_columns = int(sqrt)

        if sqrt == nb_columns:
            nb_rows = nb_columns
        elif sqrt - nb_columns < 0.5:
            nb_rows = nb_columns + 1
        else:
            nb_rows = nb_columns + 2

        fig, axs = plt.subplots(nb_rows, nb_columns, figsize=(
            10 * nb_rows, 5 * nb_columns), squeeze=False)
        for i in range(nb_rows):
            for j in range(nb_columns):
                cur_tree = i * nb_columns + j
                if cur_tree == len(best_trees):
                    break
                draw_tree(best_trees[cur_tree].genes, axs[i, j])

        plt.show()
        plt.savefig('Trees.png')

    def update_nb_inds_tasks(self, population: Population, generation: int):
        #NOTE: Hardcode here
        population.update_nb_inds_tasks([int(
            int(min((self.nb_inds_min - self.nb_inds_each_task[0])/(self.nb_generations - 1)
                * (generation - 1) + self.nb_inds_each_task[0], self.nb_inds_each_task[0]))
        )] * self.num_sub_tasks)

    def generation_step(self, population: Population, generation: int):

        # update nb_inds_tasks
        self.update_nb_inds_tasks(population=population, generation=generation)

        # create new individuals
        self.reproducer(population)

        # train and get fitness from individuals
        population.optimize()
        population.collect_fitness_info()

        # ranking
        self.ranker(population)

        # select best indivudals
        self.selector(population)

        # update info to display
        population.collect_best_info()
        
        self.update_process_bar(population, reverse=not self.is_larger_better)

    def init_params(self, population: Population, **kwargs):
        self.num_sub_tasks: int = 1
        
        self.is_larger_better = kwargs['is_larger_better']

        return {}

    def finetune(self, ind: Individual):
        ind.finetune()

    def fit(self, X: np.ndarray,
            y: np.ndarray,
            loss: Loss,
            optimzier: GradOpimizer,
            metric: Metric,
            batch_size: int = 10,
            shuffle: bool = True,
            steps_per_gen: int = 10,
            nb_generations: int = 100,
            nb_inds_each_task: int = 100,
            nb_inds_min: int = 10,
            tree_config: dict = {},
            visualize: bool = False,
            test_size: float = 0.2,
            finetune_steps: int = 5000,
            finetune_decay_lr: float = 100,
            num_workers: int = 4,
            offspring_size: float = 1.0,
            expected_generations_inqueue: int = 5000,
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

        assert X.shape[0] == y.shape[0]
        self.offspring_size= offspring_size
        self.expected_generations_inqueue= expected_generations_inqueue
        self.nb_inds_min = nb_inds_min
        self.nb_generations = nb_generations
        self.nb_inds_each_task = nb_inds_each_task
        
        self.nb_terminals = X.shape[1]

        #init multiprocessor
        initWM((1000000, max(tree_config.get('max_length'))))
        
        #init offsprings pool
        initOffspringsPool()

        self.init_params(**params, is_larger_better= metric.is_larger_better)
        
        

        with Multiprocessor(num_workers= num_workers) as multiprocessor:
            self.multiprocessor = multiprocessor
            
            # initialize population
            population = Population(
                nb_inds_tasks= self.nb_inds_each_task,
                task=Task(X, y, loss, optimzier, metric, steps_per_gen=steps_per_gen,
                        batch_size=batch_size, test_size=test_size,
                        shuffle=shuffle),
                tree_config=tree_config, num_sub_tasks=self.num_sub_tasks,
                offspring_size=offspring_size, multiprocessor= multiprocessor
            )


            # update task info for reproducer
            self.reproducer.update_task_info(
                **tree_config,
                **{attr: getattr(self, attr) for attr in self.pass_down_params},
            )

            self.pbar = GAProgressBar(
                num_iters=nb_generations, metric_name=str(metric))

            try:
                for generation in self.pbar.pbar:
                    self.generation_step(population, generation)
                else:
                    #wait for remaining jobs
                    while not self.terminated:
                        self.generation_step(population, -1)
                    
                # finetune solution
                # self.final_solution.finetune(finetune_steps= finetune_steps, decay_lr= finetune_decay_lr, verbose = True)

            except Exception as e:
                print(traceback.format_exc())
                raise e
            
            else:
                if visualize:
                    self.display_final_result(population)

            finally:
                Timer.display()
                best_trees, self.final_solution = population.get_best_trees()
            
    def predict(self, X: np.ndarray):
        return self.final_solution(X)
