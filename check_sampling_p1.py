# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from VBLCMM import *
from basis_function import *

np.random.seed(1)
output_directory = './results/check_sampling_p1/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

k = 2
p = 1
phi_func = quadratic
hyperparams = {
    'k': k,
    'sigma2':0.01,
    'lam2_0': 1.0,
    'alpha_0': np.ones(k) * 10.0,
    'mu_0': np.zeros(p),
    'tau_0': 0.25,
    'W_0': np.eye(p) * 1.0,
    'nu_0': p + 2
}

true_model = VBLCMM(hyperparams, p=p, phi_func=phi_func)

fig = plt.figure(figsize=(20,12))
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i)
    true_model.reset_params_random()
    base_instance = true_model.create_base_instance()
    base_instance.plot(ax, showMean=True)
plt.savefig(output_directory + "fig_.png")
