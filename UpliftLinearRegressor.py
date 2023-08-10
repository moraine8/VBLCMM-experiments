# coding: utf-8
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, gamma, multivariate_normal
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from util import *


class BaseUpliftLinearRegressor(BaseEstimator):
    def __init__(self, p=None, phi_func=None, sigma2=0.0):
        self.p = p
        self.phi_func = phi_func
        self.sigma2 = sigma2
        self.m = self.get_phi_dimension()
        self.b = None
        self.g = None

    def get_phi_dimension(self):
        Phi = self.phi_func(np.ones((1, self.p)))
        return len(Phi.T)

    def reset_params_gaussian(self, lam2=1.0):
        self.b = np.random.normal(0, np.sqrt(lam2), self.m)
        self.g = np.random.normal(0, np.sqrt(lam2), self.m)
        return self

    def reset_params_uniform(self, range=10):
        self.b = range * 2 * np.random.random((self.m, 1)) - range
        self.g = range * 2 * np.random.random((self.m, 1)) - range

    def create_toy_data(self, N, include_counter_factual=False, return_dataframe=False):
        if self.p > 1:
            X = np.random.multivariate_normal(np.zeros(self.p), np.eye(self.p), N)
        else:
            X = np.random.normal(0, 1.0, (N, 1))
        Phi = self.phi_func(X)
        w = np.random.randint(0, 2, N)
        y = np.array([(h(self.b + w[n] * self.g) @ v(Phi[n])).item() for n in range(N)])
        epsilon = np.random.normal(0, np.sqrt(self.sigma2), N)
        y += epsilon
        y_cf = None
        t = None
        if include_counter_factual:
            epsilon_cf = np.random.normal(0, np.sqrt(self.sigma2), N)
            y_cf = np.array([(h(self.b + (1 - w[n]) * self.g) @ v(Phi[n])).item() for n in range(N)])
            y_cf += epsilon_cf
            t = (1 - w) * (y_cf - y) + w * (y - y_cf)

        if return_dataframe:
            return self.convert_to_pandas(X, w, y, y_cf, t)
        else:
            if include_counter_factual:
                return X, w, y, y_cf, t
            else:
                return X, w, y

    def convert_to_pandas(self, X, w, y, y_cf, t):
        data = np.hstack((X, v(w), v(y)))
        if y_cf is not None and t is not None:
            data = np.hstack((data, v(y_cf), v(t)))
        columns = []
        columns_int = []
        for i in range(self.p):
            columns.append('x' + str(i + 1))
        columns.append('w')
        columns_int.append('w')
        columns.append('y')
        if y_cf is not None and t is not None:
            columns.append('y_cf')
            columns.append('t')
        df = pd.DataFrame(data=data, columns=columns)
        df[columns_int] = df[columns_int].astype('int')
        return df

    def plot(self, ax, N=50, show_mean=False):
        if self.p != 1:
            raise "Dimension Error"

        df = self.create_toy_data(N, return_dataframe=True)
        df_tmp = df[df['w'] == 0]
        x_tmp = df_tmp['x1'].to_numpy()
        y_tmp = df_tmp['y'].to_numpy()
        ax.scatter(x_tmp, y_tmp, color='tab:blue', marker='.')

        df_tmp = df[df['w'] == 1]
        x_tmp = df_tmp['x1'].to_numpy()
        y_tmp = df_tmp['y'].to_numpy()
        ax.scatter(x_tmp, y_tmp, color='tab:blue', marker='x')

        if show_mean:
            x_min = df['x1'].min()
            x_max = df['x1'].max()
            x_support = np.linspace(x_min, x_max, 100)
            y_0 = self.predict(v(x_support), w=0)
            ax.plot(x_support, y_0, color='tab:blue')
            y_1 = self.predict(v(x_support), w=1)
            ax.plot(x_support, y_1, linestyle='dashed', color='tab:blue')
        return ax

    def score(self, X_test, t_test):
        t_pred = self.predict(X_test)
        return mean_squared_error(t_pred, t_test)

    def predict(self, X, w=None):
        Phi = self.phi_func(X)
        if w == 0:
            y = Phi @ v(self.b)
        elif w == 1:
            y = Phi @ v(self.b + self.g)
        else:
            y = Phi @ v(self.g)
        return y.flatten()

    def fit(self, X, w, y, **kwargs):
        self._set_data(X, w, y)
        return self._fit(**kwargs)

    def _set_data(self, X, w, y):
        self.Phi_0 = self.phi_func(X[w == 0])
        self.Phi_1 = self.phi_func(X[w == 1])
        self.Phi = np.vstack((self.Phi_0, self.Phi_1))
        self.y_0 = y[w == 0]
        self.y_1 = y[w == 1]
        self.y = np.concatenate([self.y_0, self.y_1])
        self.w = np.sort(w)
        self.N = len(X)
        self.N_0 = len(self.Phi_0)
        self.N_1 = len(self.Phi_1)


class DoubleModelUpliftLinearRegressor(BaseUpliftLinearRegressor):
    def __init__(self, p=None, phi_func=None, sigma2=None):
        super().__init__(p=p, phi_func=phi_func, sigma2=sigma2)

    def _fit(self):
        self.b = np.linalg.pinv(self.Phi_0.T @ self.Phi_0) @ self.Phi_0.T @ self.y_0
        self.b_plus_g = np.linalg.pinv(self.Phi_1.T @ self.Phi_1) @ self.Phi_1.T @ self.y_1
        self.g = self.b_plus_g - self.b

        r_0 = self.y_0 - self.Phi_0 @ self.b
        n = self.N_0 - self.m
        if n == 0:
            n = 1.0
        sigma2 = np.sum(r_0 ** 2) / n
        if sigma2 < 0:
            sigma2 = 0
        self.sigma_0_hat = np.sqrt(sigma2)

        r_1 = self.y_1 - self.Phi_1 @ self.b_plus_g
        n = self.N_1 - self.m
        if n == 0:
            n = 1.0
        sigma2 = np.sum(r_1 ** 2) / n
        if sigma2 < 0:
            sigma2 = 0
        self.sigma_1_hat = np.sqrt(sigma2)


class JamesSteinUpliftLinearRegressor(DoubleModelUpliftLinearRegressor):
    def __init__(self, p=None, phi_func=None, sigma2=None):
        super().__init__(p=p, phi_func=phi_func, sigma2=sigma2)

    def _fit(self):
        super()._fit()
        V_inv = np.linalg.pinv((self.sigma_0_hat ** 2) * np.linalg.pinv(self.Phi_0.T @ self.Phi_0) + (self.sigma_1_hat ** 2) * np.linalg.pinv(self.Phi_1.T @ self.Phi_1))
        beta_u_d_bar = np.ones((self.m, 1)) * np.sum(self.g) / self.N_0
        beta_diff = v(self.g) - beta_u_d_bar
        self.g = ((self.m - 3) / np.asscalar(beta_diff.T @ V_inv @ beta_diff)) * beta_diff + v(self.g)


class StandardMeanSquaredErrorsUpliftLinearRegressor(DoubleModelUpliftLinearRegressor):
    def __init__(self, p=None, phi_func=None, sigma2=None):
        super().__init__(p=p, phi_func=phi_func, sigma2=sigma2)

    def _fit(self):
        super()._fit()
        XX_c = self.Phi_0.T @ self.Phi_0
        XX_t = self.Phi_1.T @ self.Phi_1
        XX = XX_t + XX_c

        V_c = (self.sigma_0_hat ** 2) * XX @ np.linalg.pinv(XX_c)
        V_t = (self.sigma_1_hat ** 2) * XX @ np.linalg.pinv(XX_t)

        bcXXbc = np.asscalar(h(self.b) @ XX @ v(self.b))
        btXXbt = np.asscalar(h(self.b_plus_g) @ XX @ v(self.b_plus_g))
        bcXXbt = np.asscalar(h(self.b) @ XX @ v(self.b_plus_g))

        a11 = btXXbt + np.trace(V_t)
        a12 = (-1) * bcXXbt
        a22 = bcXXbc + np.trace(V_c)
        A = np.array([[a11, a12], [a12, a22]])

        bcXXbu = np.asscalar(h(self.b) @ XX @ v(self.g))
        btXXbu = np.asscalar(h(self.b_plus_g) @ XX @ v(self.g))
        b = np.array([[btXXbu], [-bcXXbu]])
        alpha = np.linalg.pinv(A) @ b
        self.g = alpha[0][0] * self.b_plus_g - alpha[1][0] * self.b


class SpecialUpliftLinearRegressor(BaseUpliftLinearRegressor):
    def __init__(self, p=None, phi_func=None, sigma2=None):
        super().__init__(p=p, phi_func=phi_func, sigma2=sigma2)

    def _fit(self):
        q_c = len(self.Phi_0) / self.N
        q_t = len(self.Phi_1) / self.N
        Phi = np.vstack((self.Phi_1, self.Phi_0))
        y_tilde = np.vstack((v(self.y_0) / q_t, -v(self.y_1) / q_c))
        self.g = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y_tilde


class CorrectedUpliftLinearRegressor(BaseUpliftLinearRegressor):
    def __init__(self, p=None, phi_func=None, sigma2=None):
        super().__init__(p=p, phi_func=phi_func, sigma2=sigma2)

    def _fit(self):
        q_c = len(self.Phi_0) / self.N
        q_t = len(self.Phi_1) / self.N
        Phi = np.vstack((self.Phi_0, self.Phi_1))

        y_star = np.concatenate([(q_t / q_c) * self.y_0, (q_c / q_t) * self.y_1])
        beta_star = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ v(y_star)

        y_c_c = v(self.y_0) - self.Phi_0 @ beta_star
        y_c_t = v(self.y_1) - self.Phi_1 @ beta_star
        y_c_tilde = np.vstack((-y_c_c / q_c, y_c_t / q_t))
        self.g = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y_c_tilde
