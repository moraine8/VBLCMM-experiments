# coding: utf-8
import datetime
import csv
import os
import time
import json
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gamma, multivariate_normal, wishart
import matplotlib.pyplot as plt
from statistics import mean

from VBLCMM import *
from basis_function import *
from experiment_make_graph import make_output_directory, make_figure

# seed = np.random.randint(2**32 - 1)
seed = 2
np.random.seed(seed)

output_directory = make_output_directory('./results/experiment1/')

# experiments condition
p = 4
phi_func_name = 'linear'
phi_func = eval(phi_func_name)
N_train_start = 25
N_train_step = 5
N_train_max = 50
N_train_list = [N for N in range(N_train_start, N_train_max + 1, N_train_step)]

# save condition
condition = {
    'seed': seed,
    'p': p,
    'phi_func_name': phi_func_name,
    'sigma2': 0.25,
    'lam2_0': 0.25,
    'alpha_0': 10.0,
    'tau_0': 0.01,
    'W_0': 1.0,
    'nu_0': p + 3,
    'N_train_list': N_train_list,
    'N_test': 50,
    'Exp_num': 1000,
    'VB_init_num': 10
}
with open(output_directory + "condition.json", "w") as f:
    json.dump(condition, f, indent=2)

hyperparams_list = get_hyperparams_list(condition, 3)
true_param_index = 1
filenames = []
for hyperparams in hyperparams_list:
    filenames.append('k=' + str(hyperparams['k']) + '.csv')

# Main experiments loop
time_start = time.perf_counter()
for exp in range(1, condition['Exp_num'] + 1):
    print('exp: %d/%d' % (exp, condition['Exp_num']))
    true_model = VBLCMM(hyperparams_list[true_param_index], p, phi_func)
    true_model.reset_params_random()

    X_train, w_train, y_train = true_model.create_toy_data(N_train_list[-1])
    X_test, w_test, y_test, y_cf_test, t_test = true_model.create_toy_data(condition['N_test'], include_counter_factual=True)

    results_row_by_model = [[] for i in range(len(hyperparams_list))]
    results_row_proposal = []
    for N in N_train_list:
        model = MetaVBLCMM(hyperparams_list, p, phi_func)
        model.fit(X_train[:N], w_train[:N], y_train[:N], init_num=condition['VB_init_num'])
        # calculate score
        score_by_model = model.score_by_model(X_test, t_test)
        for l, score in enumerate(score_by_model):
            results_row_by_model[l].append(score)
        results_row_proposal.append(model.score(X_test, t_test))

    # save
    for (filename, results_row) in zip(filenames, results_row_by_model):
        with open(output_directory + filename, mode='a') as fp:
            np.savetxt(fp, np.array([results_row]), delimiter=',')
    with open(output_directory + '0_Proposal.csv', mode='a') as fp:
        np.savetxt(fp, np.array([results_row_proposal]), delimiter=',')

time_end = time.perf_counter()
print(time_end - time_start)

make_figure(output_directory)
