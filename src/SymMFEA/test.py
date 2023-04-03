

from SymMFEA.components.tree import TreeFactory, Tree
from SymMFEA.components.functions import *
from SymMFEA.components.trainer.grad_optimizer import GradOpimizer
import numpy as np
from SymMFEA.utils.visualize import draw_tree
import matplotlib.pyplot as plt 
from SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from SymMFEA.evolution.population.individual import Individual
from SymMFEA.evolution.population.population import Population
from SymMFEA.evolution.task import Task
from sklearn.datasets import make_regression
from SymMFEA.evolution.algorithms import GA
from SymMFEA.evolution.reproducer.mutation import VariableMutation, MutationList, NodeMutation
from SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from SymMFEA.components.trainer.loss import MSE
from SymMFEA.components.metrics import MAE,MAPE, R2
from SymMFEA.components.trainer.grad_optimizer import GradOpimizer
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from xgboost import XGBRegressor as XGB
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression as LNR
np.seterr(all='raise')

 

# X, y = make_regression(n_samples = 100, n_features= 10)
X, y = load_diabetes(return_X_y= True)
X = X.astype(np.float64)
y = y.astype(np.float64)

X_train, X_val = X[: 320], X[320:]
y_train, y_val = y[: 320], y[320:]
tree_config = {
    'max_length': 40,
    'max_depth': 9,
}
xgb = XGB()
gbr = GBR()
lnr = LNR()
xgb.fit(X_train, y_train)
gbr.fit(X_train, y_train)
lnr.fit(X_train, y_train)


crossover = SubTreeCrossover()
mutation = MutationList(
    [VariableMutation(),
    #  NodeMutation()
     ]
)

loss = MSE()
optimizer = GradOpimizer(1e-1)
model = GA(
    reproducer_config={
        'crossover': crossover,
        'mutation': mutation,
        'crossover_size': 1,
        'mutation_size': 1,
    }
)

model.fit(
    X = X_train, y= y_train, loss = loss,
    steps_per_gen= 2,
    nb_inds_each_task= 10,
    nb_generations= 100,
    batch_size= 100,
    nb_not_improve= 2,
    test_size = 0.2,
    optimzier=optimizer, metric =  R2(), tree_config= tree_config
)

xgb_pred = xgb.predict(X_val)
gbr_pred = gbr.predict(X_val)
lnr_pred = lnr.predict(X_val)
ga_pred = model.predict(X_val)
print('Linear: {:.2f}, sklearn.GradientBoosting: {:.2f}'.format(r2_score(y_val, lnr_pred), r2_score(y_val, gbr_pred)))
print('XGBoost: {:.2f}, SymGA: {:.2f}'.format(r2_score(y_val, xgb_pred), r2_score(y_val, ga_pred)))