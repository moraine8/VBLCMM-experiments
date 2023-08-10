# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from VBLCMM import *
from basis_function import *

np.random.seed(1)
output_directory = './results/check_sampling_p2/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

k = 2
p = 2
phi_func = linear
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
true_model.reset_params_random()

df_train = true_model.create_toy_data(100, return_dataframe=True)
X = df_train.iloc[:, 0:p].to_numpy()
w = df_train['w'].to_numpy()
y = df_train['y'].to_numpy()

x1_min = df_train['x1'].min()
x1_max = df_train['x1'].max()
x2_min = df_train['x2'].min()
x2_max = df_train['x2'].max()
y_min = df_train['y'].min()
y_max = df_train['y'].max()
w_index = []
w_index.append( df_train['w'] == 0 )
w_index.append( df_train['w'] == 1 )

fig = plt.figure(1)

# The scatter of (x1, x2)
ax = fig.add_subplot(2, 2, 1)
color = ['r','g','b','c','m','y','k']
z = []
for l in range(k):
    z.append( df_train['z'+str(l+1)] == 1 )
    sc = ax.scatter(df_train[z[l]]['x1'], df_train[z[l]]['x2'], marker='x', c=color[l])
    #sc = ax.scatter(true_model.M[l][0], true_model.M[l][1], marker='o', c=color[l])

# The scatter plot of all data
ax2 = fig.add_subplot(2, 2, 2)
sc2 = ax2.scatter(df_train[w_index[0]]['x1'], df_train[w_index[0]]['x2'], vmin=y_min, vmax=y_max, c=df_train[w_index[0]]['y'], marker='.')
sc2 = ax2.scatter(df_train[w_index[1]]['x1'], df_train[w_index[1]]['x2'], vmin=y_min, vmax=y_max, c=df_train[w_index[1]]['y'], marker='.')
plt.colorbar(sc2)

# The scatter plot of w=0
ax3 = fig.add_subplot(2, 2, 3)
sc3 = ax3.scatter(df_train[w_index[0]]['x1'], df_train[w_index[0]]['x2'], vmin=y_min, vmax=y_max, c=df_train[w_index[0]]['y'], marker='.')
plt.colorbar(sc3)

# The scatter plot of w=1
ax4 = fig.add_subplot(2, 2, 4)
sc4 = ax4.scatter(df_train[w_index[1]]['x1'], df_train[w_index[1]]['x2'], vmin=y_min, vmax=y_max, c=df_train[w_index[1]]['y'], marker='.')
plt.colorbar(sc4)

plt.savefig(output_directory + "fig_.png")
