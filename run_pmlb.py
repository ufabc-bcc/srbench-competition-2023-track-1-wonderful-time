

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
from sklearn.model_selection import train_test_split
from pmlb import fetch_data
from pmlb import regression_dataset_names
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import pandas as pd


def eval_other(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_hat = model.predict(X_val)
    return r2_score(y_val, y_hat)

def update_score(score:dict, name, r2):
    if score.get(name) is None:
        score[name] = []
    score[name].append(r2)

def eval_ds(dataset, score: dict):
    #============= Load Data ======================
    Z = dataset.values.astype(np.float32)
    X, y = Z[:, : -1], Z[:, -1]


    X = X.astype(np.float32)

    print(X.shape)
    y = y.astype(np.float32) 

    X_train, X_val, y_train, y_val = train_test_split(X, y)



    #========================= Prepare config==================

    tree_config = {
        'max_length': [80]* 2 + [30] * 2 + [10] * 5 ,
        'max_depth': 8,
        'num_columns': 1,
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
        steps_per_gen= 30,
        nb_inds_each_task= 80,
        data_sample = 1,
        nb_generations= 200,
        X_val = X_val,
        y_val = y_val,
        test_size = 0.2,
        nb_inds_min= 20,
        finetune_steps= 20,
        optimzier=optimizer, metric =  R2(), tree_config= tree_config,
        visualize= True,
        num_workers= 40,
        offspring_size= 1,
        expected_generations_inqueue= 5,
        compact= False,
        moo= True, 
        max_tree= 15000,
        trainer_config= {
            'early_stopping': 5
        },
        **SMP_configs,
    )



    #================ Other models ==================
    
    models = [
        ('XGboost',XGB(objective="reg:squarederror")),
        ('GradientBoosting', GBR()),
        ('Linear', LNR()),
        ('Logistic', LogisticRegression()),
        ('GaussianNB', GaussianNB()),
    ]
    
    for name, mod in models:
        r2 = eval_other(mod, X_train, y_train, X_val, y_val)
        print(name, r2)
        update_score(score, name, r2)

    #===================================== Predict and display result ===========================
    ga_pred = model.predict(X_val)
    r2 = r2_score(y_val, ga_pred)
    print('SymGA: {}'.format(r2))
    update_score(score, 'SymGA', r2)
    
def main():
    score = dict()
    try:
        ds_names = regression_dataset_names[:2] 
        for ds in tqdm(ds_names):
            eval_ds(fetch_data(ds), score)
    except KeyboardInterrupt:
        ...
    finally:
        result = pd.DataFrame(score)
        result.set_index(ds_names)
        result.to_csv('result.csv')
        
if __name__ == "__main__":
    main()