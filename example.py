# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt

from VBLCMM import *
from basis_function import *

np.random.seed(1)
output_directory = './results/example/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

k = 2
p = 1
m = 3
pi = np.ones(k) / k
M = np.array([[-2.0], [2.0]])
L = np.array([[1.0], [1.0]])
B = np.array([[1.0, 0, -0.5], [1.0, 0, 0.5]])
G = np.array([[0, 0, -1.0], [0, 0, 1.0]])
sigma2 = 0.1

example_model = BaseLCMM(k=k, p=p, phi_func=quadratic, pi=pi, M=M, L=L, B=B, G=G, sigma2=sigma2)
fig = plt.figure(1)
ax = fig.add_subplot(111)
example_model.plot(ax, showMean=False)
plt.savefig(output_directory + "fig_.png")