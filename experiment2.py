# coding: utf-8
import datetime
import csv
import os
import time
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gamma, multivariate_normal, wishart
import matplotlib.pyplot as plt
from statistics import mean

from VBLCMM import *
from UpliftLinearRegressor import *
from basis_function import *
from experiment_make_graph import make_output_directory, make_figure

# seed = np.random.randint(2**32 - 1)
seed = 1
np.random.seed(seed)

output_directory = make_output_directory('./results/experiment2/')

p = 4
phi_func_name = 'linear'
phi_func = eval(phi_func_name)
N_train_start = 25
N_train_step = 5
N_train_max = 50
N_train_list = [N for N in range(N_train_start, N_train_max + 1, N_train_step)]
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
    'N_test': 100,
    'Exp_num': 2,
    'VB_init_num': 2
}

with open(output_directory + "condition.json", "w") as f:
    json.dump(condition, f, indent=2)

hyperparams_list = get_hyperparams_list(condition, 3)

true_model = BaseUpliftLinearRegressor(p, phi_func, condition['sigma2'])

# compared models
compared_models = [
    {'estimator': DoubleModelUpliftLinearRegressor(p, phi_func), 'filename': '1_DoubleModel.csv'},
    {'estimator': JamesSteinUpliftLinearRegressor(p, phi_func), 'filename': '2_JamesStein.csv'},
    {'estimator': StandardMeanSquaredErrorsUpliftLinearRegressor(p, phi_func), 'filename': '3_SMSE.csv'},
    {'estimator': CorrectedUpliftLinearRegressor(p, phi_func), 'filename': '4_CorrectedUplift.csv'}
]

time_start = time.perf_counter()
for exp in range(1, condition['Exp_num'] + 1):
    print('exp: %d/%d' % (exp, condition['Exp_num']))
    true_model.reset_params_uniform(2)

    X_train, w_train, y_train = true_model.create_toy_data(N_train_list[-1])
    X_test, w_test, y_test, y_cf_test, t_test = true_model.create_toy_data(condition['N_test'], include_counter_factual=True)

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
