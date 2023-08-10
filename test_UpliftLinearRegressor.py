# coding: utf-8
import datetime
import csv
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from UpliftLinearRegressor import *
from basis_function import *


seed = np.random.randint(2**32 - 1)
# seed = 3887480892
print('seed', seed)
np.random.seed(seed)

np.set_printoptions(suppress=True)

p = 2
phi_func = linear

true_model = BaseUpliftLinearRegressor(p, linear, 0.01)
true_model.reset_params_uniform(5)

X_train, w_train, y_train = true_model.create_toy_data(100)
X_test, w_test, y_test, y_cf_test, t_test = true_model.create_toy_data(100, include_counter_factual=True)

dm_ULR = DoubleModelUpliftLinearRegressor(p, linear)
dm_ULR.fit(X_train, w_train, y_train)
print('Double: %f' % dm_ULR.score(X_test, t_test))

c_ULR = CorrectedUpliftLinearRegressor(p, linear)
c_ULR.fit(X_train, w_train, y_train)
print('Corrected: %f' % c_ULR.score(X_test, t_test))

s_ULR = SpecialUpliftLinearRegressor(p, linear)
s_ULR.fit(X_train, w_train, y_train)
print('Special: %f' % s_ULR.score(X_test, t_test))

js_ULR = JamesSteinUpliftLinearRegressor(p, linear)
js_ULR.fit(X_train, w_train, y_train)
print('JamesStein: %f' % js_ULR.score(X_test, t_test))

smse_ULR = StandardMeanSquaredErrorsUpliftLinearRegressor(p, linear)
smse_ULR.fit(X_train, w_train, y_train)
print('SMSE: %f' % smse_ULR.score(X_test, t_test))
