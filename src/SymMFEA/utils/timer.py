import time 
from termcolor import colored
from functools import wraps



class Timer:
    time_logger = {}
    
    @classmethod
    def update_time(cls, func_name, time):
        if func_name not in cls.time_logger:
            cls.time_logger[func_name] = time 
        
        else:
            cls.time_logger[func_name] += time
    

    @classmethod
    def flush(cls):
        cls.time_logger = {}
    
    
    @classmethod
    def display(cls, flush = True):
        
        argmax = ''
        m = 0
        for k, v in cls.time_logger.items():
            if v > m:
                m = v
                argmax = k  
        
        cls.time_logger['total'] = sum([t for t in cls.time_logger.values()])
        
        max_len = max([len(k) for k in cls.time_logger.keys()])
        
        display_str = '\n' + '=' * 100 + '\n'
        display_str += colored('Module', 'green') + (max_len * ' ')
        display_str += colored('  Run Time', 'green') + (7) * ' '
        display_str += colored('Percentage', 'green')
        
        
        
        
        for k, v in cls.time_logger.items():
            display_str += '\n'
            func_name = k  + (max_len - len(k) + 5) * ' ' + ':  '
            time = '{:.5f}'.format(v)
            time += (15 - len(time)) * ' ' 
            
            percent = '{:.2f}'.format(v / cls.time_logger['total'] * 100) 
            
            percent += (6 - len(percent)) * ' ' + '%'
            
            row = func_name + time + percent
            if k == argmax or k == 'total':
                row = colored(row, 'red')
                
            display_str += row  
        
        display_str += '\n' + '=' * 100 + '\n'
        
        print(display_str)
        cls.flush()
        



def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        Timer.update_time(func.__qualname__, end - start)
        return result

    return wrapper