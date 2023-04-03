from tqdm.auto import tqdm
from typing import List
from termcolor import colored
import numpy as np
class ProgressBar:
    def __init__(self, generations: int, metric_name:str = 'Obj') -> None:
        self.pbar = tqdm(range(generations))
        self.pbar.set_description(colored('GA progress', 'red'))
        self.metric_name = metric_name
    
    def update(self, objectives: List[int], reverse = False):
        argmax = np.argmax(objectives)
        
        objectives = objectives if not reverse else [-o for o in objectives]
        
        display_str = f'{self.metric_name}: '
        
        for i, o in enumerate(objectives):
            text = '{:.2f} '.format(o)
            if i == argmax:
                text = colored(text, 'green')
            display_str +=  text
            
        self.pbar.set_postfix_str(display_str)
