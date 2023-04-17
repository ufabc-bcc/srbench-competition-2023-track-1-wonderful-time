from tqdm.auto import tqdm
from typing import List
from termcolor import colored
import numpy as np


class ProgressBar:
    def __init__(self, num_iters: int, dsc: str = ' ', metric_name:str = 'Obj', **kwargs) -> None:
        self.pbar = tqdm(range(num_iters), position= 0)
        self.pbar.set_description(colored(dsc, 'red'))
        self.metric_name = metric_name
        
    def update(self, **kwargs):
        pass
        
class GAProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name:str = 'Obj', **kwargs) -> None:
        super().__init__(num_iters= num_iters, metric_name= metric_name, dsc = 'GA progress')
    
    def update(self, objectives: List[int], reverse = False, train_steps:int= 0):
        argmax = np.argmax(objectives)
        
        objectives = objectives if not reverse else [-o for o in objectives]
        
        display_str = colored(f'Train steps: {train_steps};', 'red')
        
        
        display_str += f' {self.metric_name}: '
        
        for i, o in enumerate(objectives):
            text = '{:.2f}; '.format(o)
            if i == argmax:
                text = colored(text, 'green')
            display_str +=  text
            
        self.pbar.set_postfix_str(display_str)
        
class FinetuneProgressBar(ProgressBar):
    def __init__(self, num_iters: int, metric_name: list = ['Obj'], **kwargs) -> None:
        '''
        metric_name: loss name, metric name
        '''
        super().__init__(num_iters, dsc = 'Finetune', metric_name = metric_name, **kwargs)
        
    def update(self, loss: float, metric: float, best_metric: float, reverse = False):
        display_str = ''
        metric = -metric if reverse else metric
        best_metric = -best_metric if reverse else best_metric
        for m, val in zip(self.metric_name + ['Best ' + self.metric_name[1]], [loss, metric, best_metric]):
            display_str += '{}: {:.2f}; '.format(m, val)
        
        self.pbar.set_postfix_str(colored(display_str, 'green'))