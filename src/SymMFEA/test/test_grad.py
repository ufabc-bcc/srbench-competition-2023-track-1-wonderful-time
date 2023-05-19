import sys 
import os
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-3]))
from SymMFEA.components.tree import Tree
from SymMFEA.components.functions import *
from SymMFEA.components.trainer.loss import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from .utils import *
except ImportError:
    from utils import *
from sklearn.datasets import make_classification
from SymMFEA.components.trainer.grad_optimizer import *
from SymMFEA.components.weight_manager import initWM
from sklearn.metrics import log_loss
initWM((100, 10))

#tanh(x0 * x1) + x2    
nodes = [
    Operand(0), Operand(1), Prod(), Tanh(), Operand(2), Sum(arity= 2)
]
tree = Tree(nodes=nodes, compile = True)
tree1 = Tree(nodes=tree.nodes, deepcopy= True, compile= True)
tree2 = Tree(nodes=tree.nodes, deepcopy= True, compile= True)
X, y = make_classification(n_features=3, n_redundant=0, n_repeated=0, n_informative=3)
X = X.astype(np.float64)
y = y.astype(np.float64) 


# logloss = MSE()
logloss = LogLossWithSigmoid()


def test_optimizer():
    epochs = 1000
    losses = []
    optimizer = GradOpimizer(lr=5e-3)
    for epoch in range(epochs):
        y_hat = tree(X)
        dy, loss = logloss(y, y_hat)

        losses.append(loss)
        
        optimizer.backprop(tree, dy, profile={})
        
    
    plt.plot(np.arange(epochs), losses, label='SGD')
    
    optimizer = ADAM(lr=5e-3)
    losses = []
    profile = {}

    for epoch in range(epochs):
        y_hat = tree1(X)
        dy, loss = logloss(y, y_hat)

        assert abs(loss - log_loss(y, sigmoid(y_hat))) < 1e-5
        losses.append(loss)
        
        profile = optimizer.backprop(tree1, dy, profile=profile)
    
    
    plt.plot(np.arange(epochs), losses, color= 'red', label='ADAM')
    
    optimizer = ADAM(lr=5e-3)
    losses = []

    for epoch in range(epochs):
        y_hat = tree2(X)
        dy, loss = logloss(y, y_hat)

        losses.append(loss)
        
        optimizer.backprop(tree2, dy, profile={})
    plt.plot(np.arange(epochs), losses, color= 'green', label='ADAM no profile')
    plt.legend()
    plt.savefig('optimizer_test.png')


def test_logloss():
    optimizer = ADAM(lr=5e-3)
    losses = []
    epochs = 1000
    for epoch in range(epochs):
        y_hat = tree(X)
        dy, loss = logloss(y, y_hat)

        losses.append(loss)
        
        optimizer.backprop(tree, dy, profile={})
        
    
    plt.plot(np.arange(epochs), losses)
    plt.savefig('logloss_test.png')
        

    
if __name__ == '__main__':
    test_optimizer()
