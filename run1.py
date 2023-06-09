

from src.SymMFEA.components.functions import *
import numpy as np
from src.SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from src.SymMFEA.evolution.algorithms import GA, SMP
from src.SymMFEA.evolution.reproducer.mutation import *
from src.SymMFEA.evolution.reproducer.crossover import SubTreeCrossover
from src.SymMFEA.components.trainer.loss import MSE
from src.SymMFEA.components.metrics import R2, MSE as MSE_met
from src.SymMFEA.components.trainer.grad_optimizer import GradOpimizer, ADAM
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from xgboost import XGBRegressor as XGB
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression as LNR
import time 
from src.SymMFEA.utils import stratify_train_test_split
# np.seterr(all='raise')

#============= Load Data ======================
ix = 1
Z = np.loadtxt(f"datasets/dataset_{ix}.csv", delimiter=",", skiprows=1)
X, y = Z[:, :-1], Z[:, -1]


X = X.astype(np.float32)


y = y.astype(np.float32) 

X_train, X_val, y_train, y_val = stratify_train_test_split(X, y, test_size= 0.2)



#========================= Prepare config==================

tree_config = {
    'max_length': [50]* 2 + [30] * 2 + [7] * 5,
    'max_depth': [6] * 2 + [4] * 2 + [3] * 5,
    'num_columns': 1,
}

crossover = SubTreeCrossover()
mutation = MutationList(
    [
    VariableMutation(),
     GrowTreeMutation(),
    #  PruneMutation(),
    #  NodeMutation(),
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
        # 'select_optimizing_inds': 0.5
    }
)
SMP_configs = {
    'p_const_intra': 0,
    'delta_lr': 0.1,
    'num_sub_task': 9,
}
#===================================== Fit ==========================

print(X_train.shape)

model.fit(
    X = X_train, y= y_train, loss = loss,
    X_val = X_val,
    y_val = y_val,
    steps_per_gen= 30,
    nb_inds_each_task= [15] * 4+ [30]* 5,
    data_sample = 0.8,
    nb_generations= 200,
     
    test_size = 0.33,
    nb_inds_min= [10] * 4 + [15]* 5,
    finetune_steps= 500,
    optimzier=optimizer, metric =  R2(), tree_config= tree_config,
    visualize= True,
    num_workers= 40,
    offspring_size= 5,
    expected_generations_inqueue= 5,
    compact= True,
    moo= True, 
    max_tree = 50000,
    trainer_config= {
        'early_stopping': 5
    },
    **SMP_configs,
)

#================ Other models ==================
xgb = XGB(objective="reg:squarederror")
gbr = GBR()
lnr = LNR()
s = time.time()
xgb.fit(X_train, y_train)
xgb_time = time.time() - s
gbr.fit(X_train, y_train)
gbr_time = time.time() - xgb_time - s
lnr.fit(X_train, y_train)
lnr_time = time.time() - gbr_time - xgb_time - s




#===================================== Predict and display result ===========================
xgb_pred = xgb.predict(X_val)
gbr_pred = gbr.predict(X_val)
lnr_pred = lnr.predict(X_val)
ga_pred = model.predict(X_val)
print('Linear: {:.2f}, {:.2f}s; sklearn.GradientBoosting: {:.2f}, {:.2f}s'.format(r2_score(y_val, lnr_pred), lnr_time, r2_score(y_val, gbr_pred), gbr_time))
print('XGBoost: {:.2f}, {:.2f}s; SymGA: {:.2f}'.format(r2_score(y_val, xgb_pred), xgb_time, r2_score(y_val, ga_pred)))