import sys 
import os
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-3]))
from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
from SymMFEA.components.trainer.loss import *
import numpy as np
from sklearn.datasets import make_classification
from SymMFEA.components.trainer.grad_optimizer import *
from SymMFEA.components.weight_manager import initWM
from termcolor import colored


nodes = [
    Operand(0), Operand(1), Operand(2), Sum(arity= 3), Tanh(), Operand(0), Prod(), Log(), Operand(0), AQ(), Operand(1), Valve()
]

print(f'Tree Length: {len(nodes)}')
initWM((1000, len(nodes)))
tree = Tree(nodes=nodes, compile = True)
X, y = make_classification(n_features=3, n_samples= 500, n_redundant=0, n_repeated=0, n_informative=3)
X = X.astype(np.float32)
y = y.astype(np.float32) 
optimizer = ADAM(lr=5e-3)

def benchmark():
    y_hat = tree(X)
    dy, loss = logloss(y, y_hat)    
    optimizer.backprop(tree, dy, profile={})
    
if __name__ == '__main__':
    import timeit
    print('Start benchmark backprop')
    speeds = []
    for i in range(11):
        time = timeit.timeit("benchmark()", setup="from __main__ import benchmark", number=1000)
        speed = 1000 / time 
        speeds.append(speed)
        print(f'Test {i}: ' + colored('{:.0f}'.format(speed), 'red') + ' its/s')
    
    speeds = speeds[1:]
    print(colored('Aggregated: Mean: {:.0f} its/s, std {:.0f} its/s'.format(np.mean(speeds), np.std(speeds)), 'green'))