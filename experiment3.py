# coding: utf-8
import datetime
import csv
import os
import time
import sys
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gamma, multivariate_normal, wishart
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statistics import mean

from VBLCMM import *
from UpliftLinearRegressor import *
from basis_function import *
from experiment_make_graph import make_output_directory, make_figure

#seed = np.random.randint(2**32 - 1)
seed = 1
np.random.seed(seed)

output_directory =  make_output_directory('./results/experiment3/')

df = pd.read_csv('./dataset/ihdp_npci_1.csv', header = None)
X_all = df.loc[:, 5:].values
w_all = df.loc[:,0].values
y_all = df.loc[:,1].values
y_cf_all = df.loc[:,2].values
t_all = (1 - w_all) * (y_cf_all - y_all) + w_all * (y_all - y_cf_all)
X_train, X_test, w_train, w_test, y_train, y_test, t_train, t_test = train_test_split(X_all, w_all, y_all, t_all)

N_train_max = len(X_train)
N_test = len(X_test)
N_train_list = [math.floor(N_train_max * i / 10) for i in range(4, 11, 2)]

p = len(X_train.T)
phi_func_name = 'linear'
phi_func = eval(phi_func_name)
condition = {
    'seed': seed,
    'p': p,
    'phi_func_name': phi_func_name,
    'sigma2': 0.25,
    'lam2_0': 1.0,
    'alpha_0': 10.0,
    'tau_0': 0.01,
    'W_0': 1.0,
    'nu_0': p + 3,
    'N_train_list': N_train_list,
    'N_test': N_test,
    'Exp_num': 10,
    'VB_init_num': 30
}

with open(output_directory + "condition.json", "w") as f:
    json.dump(condition, f, indent=2)

hyperparams_list = get_hyperparams_list(condition, 3)

# compared models
compared_models = [
    {'estimator': DoubleModelUpliftLinearRegressor(p, phi_func), 'filename': '1_DoubleModel.csv'},
    {'estimator': JamesSteinUpliftLinearRegressor(p, phi_func), 'filename': '2_JamesStein.csv'},
    {'estimator': StandardMeanSquaredErrorsUpliftLinearRegressor(p, phi_func), 'filename': '3_SMSE.csv'},
    #{'estimator': CorrectedUpliftLinearRegressor(p, phi_func), 'filename': '4_CorrectedUplift.csv'}
]

time_start = time.perf_counter()
for exp in range(1, condition['Exp_num'] + 1):
    print('exp: %d/%d' % (exp, condition['Exp_num']))
    X_train, X_test, w_train, w_test, y_train, y_test, t_train, t_test = train_test_split(X_all, w_all, y_all, t_all)
    for model in compared_models:
        score_row = []
        for N in N_train_list:
            model['estimator'].fit(X_train[:N], w_train[:N], y_train[:N])
            score_row.append(model['estimator'].score(X_test, t_test))
        with open(output_directory + model['filename'], mode='a') as fp:
            np.savetxt(fp, np.array([score_row]), delimiter=',')

    score_row = []
    for N in N_train_list:
        proposal_model = MetaVBLCMM(hyperparams_list, p=p, phi_func=phi_func)
        proposal_model.fit(X_train[:N], w_train[:N], y_train[:N], init_num=condition['VB_init_num'])
        score_row.append(proposal_model.score(X_test, t_test))
    with open(output_directory + '0_Proposal.csv', mode='a') as proposal_fp:
        np.savetxt(proposal_fp, np.array([score_row]), delimiter=',')


time_end = time.perf_counter()
print(time_end - time_start)

make_figure(output_directory)