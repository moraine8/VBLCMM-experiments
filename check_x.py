# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from VBLCMM import *
from basis_function import *

np.random.seed(1)
output_directory = './results/check_x/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

k = 3
p = 2
phi_func = linear
hyperparams = {
    'k': k,
    'sigma2':0.01,
    'lam2_0': 1.0,
    'alpha_0': np.ones(k) * 10.0,
    'mu_0': np.zeros(p),
    'tau_0': 0.1,
    'W_0': np.eye(p) * 1.0,
    'nu_0': p + 3
}

true_model = VBLCMM(hyperparams, p=p, phi_func=phi_func)

fig = plt.figure(figsize=(20,12))
for i in range(1, 10):
    print(i)
    true_model.reset_params_random()
    X, w, y = true_model.create_toy_data(1000)
    ax = fig.add_subplot(3, 3, i)
    H = ax.hist2d(X.T[0], X.T[1], bins=30, cmap=cm.jet)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig.colorbar(H[3],ax=ax)
plt.savefig(output_directory + "fig_.png")
