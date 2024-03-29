from tqdm import tqdm
from typing import List, Union, Iterable
from termcolor import colored
import numpy as np
from ..components.metrics import Metric

class ProgressBar:
    def __init__(self, num_iters: Union[int, Iterable], dsc: str = ' ', metric_name:str = 'Obj', leave= True, position = 0, miniters:int=None, **kwargs) -> None:
        iterable = range(num_iters) if isinstance(num_iters, int) else num_iters

        self.pbar = tqdm(iterable, leave= leave, position= position, colour='blue', miniters= miniters)
        self.pbar.set_description(colored(dsc, 'red'))
        self.metric_name = metric_name
        
    def update_what_iam_doing(self, **kwargs):
        ...
        
    def update(self, **kwargs):
        ...
    
    def __enter__(self):
        return self, self.pbar.__enter__()
    
    def __exit__(self, *args, **kwargs):
        
        self.pbar.__exit__(*args, **kwargs)
        
    
    def set_finished(self):
        self.pbar.colour= 'green'
        self.pbar.refresh()
        
        
class GAProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name:str = 'Obj', **kwargs) -> None:
        super().__init__(num_iters= num_iters, metric_name= metric_name, dsc = 'GA progress')
    
    def update(self, objectives: List[int], reverse = False, train_steps:int= 0, nb_inqueue: int =0, processed:int = 0, multiprocessor_time:float =0, cur_met = None):
        argmax = np.argmax(objectives)
        
        objectives = objectives if not reverse else [-o for o in objectives]
        
        display_str = '' if cur_met is None else colored('Val {}: {:.2f}, '.format(self.metric_name, cur_met), 'green')
        
        display_str += colored('Train steps: {:,}, In queue: {:,}, Processed: {:,}, Processor time: {:.2f} s;'.format(train_steps, nb_inqueue, processed, multiprocessor_time), 'red')
        
        
        display_str += f' {self.metric_name}: '
        
        for i, o in enumerate(objectives):
            text = '{:.2f}; '.format(o)
            if i == argmax:
                text = colored(text, 'green')
            display_str +=  text
            
        self.pbar.set_postfix_str(display_str)
        self.pbar.refresh()
        
    def set_waiting(self):
        self.pbar.set_description(colored('Waiting for queue', 'red'))
    
    def set_finished(self):
        super().set_finished()
        self.pbar.set_description(colored('GA finished', 'green'))
        self.pbar.refresh()
        
        
class FinetuneProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name: list = ['Obj'], **kwargs) -> None:
        '''
        metric_name: loss name, metric name
        '''
        super().__init__(num_iters, dsc = 'Candidates', metric_name = metric_name, leave=False, position= 1, miniters= int(num_iters / 10),**kwargs)
        
    def update(self, loss: float, metric: float, best_metric: float, reverse = False):
        display_str = ''
        
        to_display_met = metric 
        to_display_best_met = best_metric
        
        metric = -metric if reverse else metric
        best_metric = -best_metric if reverse else best_metric
        for m, val in zip(self.metric_name + ['Best ' + self.metric_name[1]], [loss, to_display_met, to_display_best_met]):
            display_str += '{}: {:.2f}; '.format(m, val)
        
        self.pbar.set_postfix_str(colored(display_str, 'green'))
        self.pbar.refresh()
        
        
class CandidateFinetuneProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric: Metric, **kwargs) -> None:
        super().__init__(num_iters, metric_name= str(metric), dsc= 'Finetuning candidates', **kwargs)
        self.curbest = None
        self.best_idx = None
        self.compare = metric.is_better

            
    def update(self, metric: float, idx:int , reverse= False):
        metric = -metric if reverse else metric
        if self.compare(metric, self.curbest):
            self.curbest= metric
            self.best_idx = idx
                    
        display_str = 'Best {}: {:.2f}; '.format(self.metric_name, self.curbest)
        
        
        self.pbar.set_postfix_str(colored(display_str, 'green'))
        self.pbar.refresh()
        
class SimplificationProgresBar(ProgressBar):
    def __init__(self, simplification_list, **kwargs) -> None:
        
        super().__init__(num_iters = simplification_list, dsc = 'Simplifying expression', metric_name= 'Number of nodes',**kwargs)
    
    
    def update_what_iam_doing(self, symplification,  **kwargs):
        self.pbar.set_description(colored(f'Doing symplification: {symplification}', 'red'))
    
    def update(self, number_of_nodes):
        display_str = 'Current nb of nodes: {:,}; '.format(number_of_nodes)
        self.pbar.set_postfix_str(colored(display_str, 'green'))
        self.pbar.refresh()
        
    def set_finished(self):
        super().set_finished()
        self.pbar.set_description(colored('Symplification finished', 'green'))
        self.pbar.refresh()