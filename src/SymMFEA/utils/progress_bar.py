from tqdm.auto import tqdm
from typing import List
from termcolor import colored
import numpy as np


class ProgressBar:
    def __init__(self, num_iters: int, dsc: str = ' ', metric_name:str = 'Obj', leave= True, position = 0, **kwargs) -> None:
        self.pbar = tqdm(range(num_iters), leave= leave, position= position)
        self.pbar.set_description(colored(dsc, 'red'))
        self.metric_name = metric_name
        
    def update(self, **kwargs):
        ...
    
    def __enter__(self):
        return self, self.pbar.__enter__()
    
    def __exit__(self, *args, **kwargs):
        
        self.pbar.__exit__(*args, **kwargs)
    
        
        
class GAProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name:str = 'Obj', **kwargs) -> None:
        super().__init__(num_iters= num_iters, metric_name= metric_name, dsc = 'GA progress')
    
    def update(self, objectives: List[int], reverse = False, train_steps:int= 0, in_queue: int =0, processed:int = 0):
        argmax = np.argmax(objectives)
        
        objectives = objectives if not reverse else [-o for o in objectives]
        
        display_str = colored(f'Train steps: {train_steps}, In queue: {in_queue}, Processed: {processed};', 'red')
        
        
        display_str += f' {self.metric_name}: '
        
        for i, o in enumerate(objectives):
            text = '{:.2f}; '.format(o)
            if i == argmax:
                text = colored(text, 'green')
            display_str +=  text
            
        self.pbar.set_postfix_str(display_str)
        
    def set_waiting(self):
        self.pbar.set_description('Waiting for individuals in queue')
    
    def set_finished(self):
        self.pbar.set_description('GA finished')
        
class FinetuneProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name: list = ['Obj'], **kwargs) -> None:
        '''
        metric_name: loss name, metric name
        '''
        super().__init__(num_iters, dsc = 'Candidates', metric_name = metric_name, leave=False, position= 1, **kwargs)
        
    def update(self, loss: float, metric: float, best_metric: float, reverse = False):
        display_str = ''
        metric = -metric if reverse else metric
        best_metric = -best_metric if reverse else best_metric
        for m, val in zip(self.metric_name + ['Best ' + self.metric_name[1]], [loss, metric, best_metric]):
            display_str += '{}: {:.2f}; '.format(m, val)
        
        self.pbar.set_postfix_str(colored(display_str, 'green'))
        
        
class CandidateFinetuneProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name: str = 'Obj', **kwargs) -> None:
        super().__init__(num_iters, metric_name= metric_name, dsc= 'Finetuning candidates', leave= True, **kwargs)
        self.curbest = 0
        self.best_idx = None
        
    def compare(self, metric:float, reverse:bool):
        if not reverse:
            return metric > self.curbest 
        else:
            return metric < self.curbest   
        
    
    def update(self, metric: float, idx:int , reverse= False):
        metric = -metric if reverse else metric
        if self.compare(metric, reverse):
            self.curbest= metric
            self.best_idx = idx
                    
        display_str = 'Best {}: {:.2f}; '.format(self.metric_name, self.curbest)
        
        
        self.pbar.set_postfix_str(colored(display_str, 'green'))
    