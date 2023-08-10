# coding: utf-8
import datetime
import csv
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from VBLCMM import *
from basis_function import *

mode = 1 # 1: VBLCMM, 2: RandomInitVBLCMM, 3: MetaVBLCMM

seed = np.random.randint(2**32 - 1)
print('seed', seed)
np.random.seed(seed)

np.set_printoptions(suppress=True)

p = 2
condition = {
    'seed': seed,
    'p': p,
    'phi_func_name': 'linear',
    'sigma2': 0.25,
    'lam2_0': 1.0,
    'alpha_0': 10.0,
    'tau_0': 0.01,
    'W_0': 1.0,
    'nu_0': p + 3,
    'VB_init_num': 10
}
phi_func = eval(condition['phi_func_name'])

hyperparams_list = get_hyperparams_list(condition, 3)

true_param_index = 1
true_model = VBLCMM(hyperparams_list[true_param_index], p=p, phi_func=phi_func)
true_model.reset_params_random()

N_train = 5
N_test = 100

X_train, w_train, y_train = true_model.create_toy_data(N_train)
X_test, w_test, y_test, y_cf_test, t_test = true_model.create_toy_data(N_test, include_counter_factual=True)

if mode == 1:
    VBLCMM = VBLCMM(hyperparams_list[2], p=p, phi_func=phi_func)
    VBLCMM.fit_debug(X_train, w_train, y_train)
    score = VBLCMM.score(X_test, t_test)
    print(score)
elif mode == 2:
    rivurl = RandomInitVBLCMM(VBLCMM(hyperparams_list[2],p=p, phi_func=phi_func))
    rivurl.fit(X_train, w_train, y_train, init_num=condition['VB_init_num'])
    score = rivurl.best_estimator_.score(X_test, t_test)
    print(score)
    fig = plt.figure(1)
    ax = fig.add_subplot()
    rivurl.show_VLBs(ax)
    plt.show()
elif mode == 3:
    model = MetaVBLCMM(hyperparams_list, p=p, phi_func=phi_func)
    model.fit(X_train, w_train, y_train, init_num=condition['VB_init_num'])
    score_by_model = model.score_by_model(X_test, t_test)
    for l, score in enumerate(score_by_model):
        print("l=%d, score: %f, q(l)=%f" % (l, score, model.q_k[l]))
    score = model.score(X_test, t_test)
    print("all, score: %f" % score)
    fig = plt.figure(1)
    ax = fig.add_subplot()
    model.show_VLBs(ax)
    plt.show()

