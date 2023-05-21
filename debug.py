

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

# np.seterr(all='raise')

#============= Load Data ======================
ix = 2
Z = np.loadtxt(f"datasets/dataset_{ix}.csv", delimiter=",", skiprows=1)
X, y = Z[:, :-1], Z[:, -1]
# X, y = load_diabetes(return_X_y= True)


X = X.astype(np.float64)


y = y.astype(np.float64) 

print(X.shape)
train_size = int(0.8 * X.shape[0])
X_train, X_val, y_train, y_val = stratify_train_test_split(X, y, test_size= 0.2)






#========================= Prepare config==================

tree_config = {
    'max_length': [50]* 2 + [30] * 2 + [7] ,
    'max_depth': [9] * 2 + [7] * 2 + [3],
    'num_columns': [1] + [0.7] * 6 + [0.4],
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
    'num_sub_task': 5,
}
#===================================== Fit ==========================
model.fit(
    X = X_train, y= y_train, loss = loss,
    steps_per_gen= 2,
    nb_inds_each_task= [15] * 4+ [30],
    data_sample = 0.5,
    nb_generations= 5,
     
    test_size = 0.33,
    nb_inds_min= [10] * 4 + [15],
    finetune_steps= 500,
    optimzier=optimizer, metric =  R2(), tree_config= tree_config,
    visualize= True,
    num_workers= 24,
    offspring_size= 2,
    expected_generations_inqueue= 5,
    compact= True,
    moo= True, 
    trainer_config= {
        'early_stopping': 3
    },
    **SMP_configs,
)
