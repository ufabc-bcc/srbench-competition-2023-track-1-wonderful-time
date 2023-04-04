

from src.SymMFEA.components.functions import *
from src.SymMFEA.components.trainer.grad_optimizer import GradOpimizer
import numpy as np
from src.SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from src.SymMFEA.evolution.algorithms import GA
from src.SymMFEA.evolution.reproducer.mutation import VariableMutation, MutationList, NodeMutation
from src.SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from src.SymMFEA.components.trainer.loss import MSE
from src.SymMFEA.components.metrics import R2
from src.SymMFEA.components.trainer.grad_optimizer import GradOpimizer
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from xgboost import XGBRegressor as XGB
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression as LNR
np.seterr(all='raise')

ix = 1
Z = np.loadtxt(f"datasets/dataset_{ix}.csv", delimiter=",", skiprows=1)
X, y = Z[:, :-1], Z[:, -1]
# X, y = load_diabetes(return_X_y= True)


X = X.astype(np.float64)
y = y.astype(np.float64)

print(X.shape)
train_size = int(0.8 * X.shape[0])
X_train, X_val = X[: train_size], X[train_size:]
y_train, y_val = y[: train_size], y[train_size:]
tree_config = {
    'max_length': 30,
    'max_depth': 6,
}
xgb = XGB(objective="reg:squarederror")
gbr = GBR()
lnr = LNR()
xgb.fit(X_train, y_train)
gbr.fit(X_train, y_train)
lnr.fit(X_train, y_train)


crossover = SubTreeCrossover(1)
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
        'crossover_size': 0.5,
        'mutation_size': 1,
    },
    selector_config={
        'select_optimizing_inds': 0.5
    }
)

model.fit(
    X = X_train, y= y_train, loss = loss,
    steps_per_gen= 3,
    nb_inds_each_task= 20,
    nb_generations= 100,
    batch_size= 2000,
    nb_not_improve= 3,
    test_size = 0.2,
    optimzier=optimizer, metric =  R2(), tree_config= tree_config,
    visualize= True,
)

xgb_pred = xgb.predict(X_val)
gbr_pred = gbr.predict(X_val)
lnr_pred = lnr.predict(X_val)
ga_pred = model.predict(X_val)
print('Linear: {:.2f}, sklearn.GradientBoosting: {:.2f}'.format(r2_score(y_val, lnr_pred), r2_score(y_val, gbr_pred)))
print('XGBoost: {:.2f}, SymGA: {:.2f}'.format(r2_score(y_val, xgb_pred), r2_score(y_val, ga_pred)))