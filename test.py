

from src.SymMFEA.components.functions import *
import numpy as np
from src.SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from src.SymMFEA.evolution.algorithms import GA, SMP
from src.SymMFEA.evolution.reproducer.mutation import *
from src.SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from src.SymMFEA.components.trainer.loss import MSE
from src.SymMFEA.components.metrics import R2
from src.SymMFEA.components.trainer.grad_optimizer import GradOpimizer, ADAM
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from xgboost import XGBRegressor as XGB
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression as LNR
# np.seterr(all='raise')

ix = 2
Z = np.loadtxt(f"datasets/dataset_{ix}.csv", delimiter=",", skiprows=1)
X, y = Z[:, :-1], Z[:, -1]
# X, y = load_diabetes(return_X_y= True)


X = X.astype(np.float64)


y = y.astype(np.float64) 

print(X.shape)
train_size = int(0.8 * X.shape[0])
X_train, X_val = X[: train_size], X[train_size:]
y_train, y_val = y[: train_size], y[train_size:]
# tree_config = {
#     'max_length': [100]* 2 + [60] * 3 + [30] * 5,
#     'max_depth': [9] * 2 + [7] * 3 + [5] * 5,
# }

tree_config = {
    'max_length': [50]* 2 + [30] * 2 + [7] * 3 + [2] * 3,
    'max_depth': [9] * 2 + [7] * 2 + [3] * 3 + [2] * 3,
    'num_columns': [0.7] * 7 + [0.4] * 3,
}
xgb = XGB(objective="reg:squarederror")
gbr = GBR()
lnr = LNR()
xgb.fit(X_train, y_train)
gbr.fit(X_train, y_train)
lnr.fit(X_train, y_train)


crossover = SubTreeCrossover(2)
mutation = MutationList(
    [
    VariableMutation(),
     GrowTreeMutation(2),
     ]
)

loss = MSE()
optimizer = ADAM(1e-2, weight_decay= 1e-5)
model = SMP(
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


SMP_configs = {
    'p_const_intra': 0,
    'delta_lr': 0.1,
    'num_sub_task': 10,
}

model.fit(
    X = X_train, y= y_train, loss = loss,
    steps_per_gen= 5,
    nb_inds_each_task= 20,
    nb_generations= 100,
    batch_size= 2000,
    nb_not_improve= 5,
    test_size = 0.2,
    nb_inds_min= 10,
    finetune_steps= 1000,
    optimzier=optimizer, metric =  R2(), tree_config= tree_config,
    visualize= True,
    **SMP_configs,
)

xgb_pred = xgb.predict(X_val)
gbr_pred = gbr.predict(X_val)
lnr_pred = lnr.predict(X_val)
ga_pred = model.predict(X_val)
print('Linear: {:.2f}, sklearn.GradientBoosting: {:.2f}'.format(r2_score(y_val, lnr_pred), r2_score(y_val, gbr_pred)))
print('XGBoost: {:.2f}, SymGA: {:.2f}'.format(r2_score(y_val, xgb_pred), r2_score(y_val, ga_pred)))