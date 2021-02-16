import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
#from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score


from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(111, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X



from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform, loguniform
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier




param_rdm_forest = {'bootstrap': [True, False],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10, 15, 100],
            'n_estimators': sp_randint(10,2000),
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None] ,
                    'verbose':[True],
                    'n_jobs':[-1]
                   }

parameters_svm = {
    'C': loguniform(a=0.01,b=5),
    'kernel': ['rbf', 'poly'],
    'gamma': loguniform(a=0.01,b=5),
    'verbose':[True],
    'max_iter':[100],
    'probability':[True]

    }


param_log_reg = {'C':sp_uniform(loc=0, scale=4),
                     'penalty':['l2', 'l1'],
                    'solver':['saga'], 
                 'tol':loguniform(a=0.001,b=1), 
                 'max_iter':loguniform(a=150,b=750),
                              'random_state':[0],
                    'n_jobs':[-1]}


param_dec_tree = {"max_depth": [1,5,10,50,100,500, None],
              "max_features": sp_randint(1, 9),
              "min_samples_leaf": sp_randint(1, 9),
              "criterion": ["gini", "entropy"]}


params_skorch = {
    'lr':  loguniform(a=0.001,b=5),
    'max_epochs': [10,20],
    'module__num_units': [10, 20],
    'module':[MyModule]
}


params_lgbm = {'learning_rate': [np.random.uniform(0, 1)],
            'boosting_type': [np.random.choice(['gbdt', 'dart'])],
            'metric' :['mae'],
            'sub_feature' : [np.random.uniform(0, 1)],
            'min_data' : [np.random.randint(10, 100)],
            'max_depth' : [np.random.randint(5, 200)],
            'num_leaves': sp_randint(6, 50), 
            'min_child_samples': sp_randint(100, 500), 
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample': sp_uniform(loc=0.2, scale=0.8), 
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha': loguniform(a=0.1,b=100),
            'reg_lambda': loguniform(a=0.1,b=100)}

param_xgboost = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': loguniform(a=0.001,b=5),
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': loguniform(a=0.5,b=10),
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': loguniform(a=0.1,b=100),
        'n_estimators': [100,150,200],
        'n_jobs':[-1]}


parameter_mlp = {
    'hidden_layer_sizes': [(10,30,10),(20,10,10),(30,10,10),(30,10),(20 )],
    'activation': ['relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0005,0.005 ,0.05,0.5,5],
    'learning_rate': ['constant','adaptive'],
}


clfs = [RandomForestClassifier,
        LogisticRegression,
       DecisionTreeClassifier,
        lgb.LGBMClassifier,
        XGBClassifier,
        MLPClassifier,
        SVC
       ]
clfs_names = [  'RandomForestClassifier',
        'LogisticRegression',
       'DecisionTreeClassifier',
        "lgb_LGBMClassifier",
        'XGBClassifier',
        'MLPClassifier',
        'SVC'
       ]

pars = [param_rdm_forest, param_log_reg, param_dec_tree,\
        params_lgbm, param_xgboost, parameter_mlp,parameters_svm]

clfs_names_dict = dict(zip(clfs_names,clfs))

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')





def find_threshold_1(y_true_th,y_proba_th, metric_1, metric_2, min_metric_2= 0.05,maximize_metric_2 = False):
    
    min_true_for_metric_1 = y_true_th.sum()*min_metric_2 #5%-7%
    
    y_proba_cum = pd.DataFrame([y_proba_th,y_true_th],index=['y_proba_th','y_true_th']).T\
        .sort_values('y_proba_th',ascending = False)
    y_proba_cum['cumulative'] = y_proba_cum.y_true_th.cumsum()
    
    min_threshold = y_proba_cum.query(f'cumulative >= {min_true_for_metric_1}').iloc[0].y_proba_th
    metric = metric_1(y_true_th,y_proba_th>min_threshold)
    return min_threshold,metric
    
    display(y_proba_cum)
    print(min_true_for_metric_1,y_proba_cum.query(f'y_proba_th == {min_threshold}'))
    thresholds = np.arange(0, 1, 0.001)
    
    #maximizar 
    # precisão @ 5% recall
    # evaluate each threshold. We are trying to optimize the first metric, guaranteeing a maximum of 0.4 of the second
    metric_1 = recall_s
    metric_2 = fpr
    
    scores = np.array([[metric_1(y_true_th, to_labels(y_proba_th, t)) , metric_2(y_true_th, to_labels(y_proba_th, t))]\
              for t in thresholds])
    if maximize_metric_2:
        scores_guaranteed_min = scores[:,0]*(scores[:,1]>=min_metric_2)
    else:
        scores_guaranteed_min = scores[:,0]*(scores[:,1]<=min_metric_2)
    #print(scores_guaranteed_min)
    #print(scores_guaranteed_min)
    ix = np.argmax(scores_guaranteed_min)
    th = thresholds[ix]
    metric = scores_guaranteed_min[ix]
    print('Threshold=%.3f, Metric=%.5f' % (th, metric))
     
    #plt.plot(thresholds, scores[:,0])
    #plt.plot(thresholds, scores[:,1])
    #plt.show()
    
    return th,metric



def find_threshold_2(y_true_th,y_proba_th, metric_1=precision_score, metric_2=recall_score,\
                   min_metric_2= 0.05,maximize_metric_1 = False):
    
    min_true_for_metric_1 = y_true_th.sum()*min_metric_2

    y_proba_cum = pd.DataFrame([y_proba_th,y_true_th],index=['y_proba_th','y_true_th']).T\
        .sort_values('y_proba_th',ascending = False)
    y_proba_cum['cumulative'] = y_proba_cum.y_true_th.cumsum()
    #display(y_proba_cum)
    min_threshold = y_proba_cum.query(f'cumulative >= {min_true_for_metric_1}').iloc[0].y_proba_th
    metric = metric_1(y_true_th,y_proba_th>min_threshold)
    
    return min_threshold,metric













