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
from sklearn.linear_model import LinearRegression
initWM((100, 10))

#x0 + x1 + x2    
nodes = [
    Operand(0), Operand(1), Operand(2), Sum(arity= 3)
]
tree = Tree(nodes=nodes, compile = True)
tree1 = Tree(nodes=tree.nodes, deepcopy= True, compile= True)
tree2 = Tree(nodes=tree.nodes, deepcopy= True, compile= True)
tree3 = Tree(nodes=tree.nodes, deepcopy= True, compile= True)
X, y = make_classification(n_features=3, n_redundant=0, n_repeated=0, n_informative=3)
X = X.astype(np.float32)
y = y.astype(np.float32) 


# logloss = MSE()
logloss = LogLossWithSigmoid()


def test_optimizer():
    epochs = 1000
    losses = []
    
    #============================SGD===================
    optimizer = GradOpimizer(lr=5e-3)
    for epoch in range(epochs):
        y_hat = tree(X)
        dy, loss = logloss(y, y_hat)
        if epoch > 0:
            assert loss - losses[-1] < 1e-3
        losses.append(loss)
        
        optimizer.backprop(tree, dy, profile={})
    
    sgd_loss = np.min(losses)
    
    plt.plot(np.arange(epochs), losses, label='SGD')
    
    
    #============================ADAM no profile===================
    optimizer = ADAM(lr=5e-3)
    losses = []
    profile = {}

    for epoch in range(epochs):
        y_hat = tree1(X)
        dy, loss = logloss(y, y_hat)

        assert abs(loss - log_loss(y, sigmoid(y_hat))) < 1e-3
        
        if epoch > 0:
            assert loss - losses[-1] < 1e-3
        losses.append(loss)
        
        profile = optimizer.backprop(tree1, dy, profile=profile)
    
    no_profile = np.min(losses)
    
    
    plt.plot(np.arange(epochs), losses, color= 'red', label='ADAM')
    
    
    
    #============================ADAM with profile===================    
    optimizer = ADAM(lr=5e-3)
    losses = []

    for epoch in range(epochs):
        y_hat = tree2(X)
        dy, loss = logloss(y, y_hat)
        if epoch > 0:
            assert loss - losses[-1] < 1e-3
        losses.append(loss)
        
        optimizer.backprop(tree2, dy, profile={})
    plt.plot(np.arange(epochs), losses, color= 'green', label='ADAM no profile')
    
    adam_loss = np.min(losses)
    
    
    
    lnr = LinearRegression()
    lnr.fit(X, y)
    y_hat = lnr.predict(X)
    dy, loss = logloss(y, y_hat)
    
    
    
    plt.axhline(y=loss, linestyle='dashed', label ='linear with mse', color = 'purple')
    plt.legend()
    plt.savefig('optimizer_test.png')
    plt.clf()
    
    assert adam_loss < sgd_loss
    assert no_profile < sgd_loss


def test_logloss():
    optimizer = GradOpimizer(lr=5e-3)
    losses = []
    epochs = 1000
    for epoch in range(epochs):
        y_hat = tree3(X)
        dy, loss = logloss(y, y_hat)
        
        if epoch > 0:
            assert loss - losses[-1] < 1e-3
        losses.append(loss)
        
        
        optimizer.backprop(tree3, dy, profile={})
        
    
    plt.plot(np.arange(epochs), losses)
    plt.savefig('logloss_test.png')
        

    
if __name__ == '__main__':
    test_optimizer()
    test_logloss()
