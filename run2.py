

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
import time 
from src.SymMFEA.utils import stratify_train_test_split
from sklearn.datasets import load_diabetes

# np.seterr(all='raise')

#============= Load Data ======================
X, y = load_diabetes(return_X_y= True)



X = X.astype(np.float32)


y = y.astype(np.float32) 

# X_train, X_val, y_train, y_val = stratify_train_test_split(X, y, test_size= 0.2)
X_train, y_train = X[:350], y[:350]
X_val, y_val = X[350:], y[350:]



#========================= Prepare config==================

tree_config = {
    'max_length': [100]* 2 + [40] * 2 + [10] * 5 ,
    'max_depth': 10,
    'max_depth': 10,
    'num_columns': 0.8,
}

crossover = SubTreeCrossover()
mutation = MutationList(
    [
    VariableMutation(),
     GrowTreeMutation(),
     PruneMutation()
     ]
)

loss = MSE()
optimizer = ADAM(1e-2, weight_decay= 1e-5)
model = SMP(
    reproducer_config={
        'crossover': crossover,
        'mutation': mutation,
    },
    selector_config={
        # 'select_optimizing_inds': 0.5
    }
)
SMP_configs = {
    'p_const_intra': 0,
    'delta_lr': 0.05,
    'num_sub_task': 9,
}
#===================================== Fit ==========================
model.fit(
    X = X_train, y= y_train, loss = loss,
    steps_per_gen= 50,
    nb_inds_each_task= [80] * 9,
    data_sample = 1,
    nb_generations= 200,
    X_val = X_val,
    y_val = y_val,
    test_size = 0.2,
    nb_inds_min= 20,
    finetune_steps= 1000,
    optimzier=optimizer, metric =  R2(), tree_config= tree_config,
    visualize= True,
    num_workers= 40,
    offspring_size= 1,
    expected_generations_inqueue= 5,
    compact= True,
    moo= True, 
    max_tree= 100000,
    trainer_config= {
        'early_stopping': 10
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