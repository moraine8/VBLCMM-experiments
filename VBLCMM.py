# coding: utf-8
import csv
import copy
import sys
from math import lgamma
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import numpy.linalg as LA
import pandas as pd
from scipy.stats import multivariate_normal, wishart
from scipy.special import psi
from sklearn.metrics import mean_squared_error

from util import *

lgamma_u = np.frompyfunc(lgamma, 1, 1)
ln2pi = np.log(2) + np.log(np.pi)


def ln_C(alpha):
    return lgamma_u(np.sum(alpha)) - np.sum(lgamma_u(alpha))


def ln_B(W, nu):
    D = len(W)
    return (-0.5 * nu) * LA.slogdet(W)[1] - ( (0.5 * nu * D) * np.log(2) + (0.25*D*(D-1)) * np.log(np.pi) + np.sum( lgamma_u( np.array([ 0.5*(nu-i) for i in range(D)]) ) ) )


def get_hyperparams_list(condition, k_max):
    hyperparams_list = []
    for l in range(1, k_max + 1):
        hyperparams_list.append({
            'k': l,
            'sigma2': condition['sigma2'],
            'lam2_0': condition['lam2_0'],
            'alpha_0': np.ones(l) * condition['alpha_0'],
            'mu_0': np.zeros(condition['p']),
            'tau_0': condition['tau_0'],
            'W_0': np.eye(condition['p']) * condition['W_0'],
            'nu_0': condition['nu_0']
        })
    return hyperparams_list


# Latent Class-based Mixture Model
class BaseLCMM(BaseEstimator):
    def __init__(self, k=None, p=None, phi_func=None, pi=None, M=None, L=None, B=None, G=None, sigma2=None):
        self.k = k
        self.p = p
        self.phi_func = phi_func
        self.pi = pi
        self.M = M
        self.L = L
        self.B = B
        self.G = G
        self.sigma2 = sigma2
        self.m = self.get_phi_dimension()

    def get_phi_dimension(self):
        Phi = self.phi_func(np.ones((1, self.p)))
        return len(Phi.T)

    def show_params(self):
        print('B', self.B)
        print('G', self.G)
        print('pi', self.pi)
        print('M', self.M)
        print('L', self.L)

    def create_toy_data(self, N, include_counter_factual=False, return_dataframe=False):
        # p(z|pi)
        z = np.random.choice(list(range(self.k)), N, p=self.pi)
        z = np.eye(self.k)[z]  # to one-hot vectors
        # p(x|M,L)
        X = np.zeros((N, self.p))
        if self.p > 1:
            for n in range(N):
                for l in range(self.k):
                    X[n] += z[n][l] * np.random.multivariate_normal(self.M[l], LA.inv(self.L[l]))
        else:
            for n in range(N):
                for l in range(self.k):
                    X[n] += z[n][l] * np.random.normal(self.M[l], 1.0 / self.L[l])
        Phi = self.phi_func(X)
        w = np.random.randint(0, 2, N)
        y = np.array([(h(z[n]) @ (self.B + w[n] * self.G) @ v(Phi[n])).item() for n in range(N)])
        epsilon = np.random.normal(0, np.sqrt(self.sigma2), N)
        y += epsilon
        y_cf = None
        t = None
        if include_counter_factual:
            epsilon_cf = np.random.normal(0, np.sqrt(self.sigma2), N)
            y_cf = np.array([(h(z[n]) @ (self.B + (1 - w[n]) * self.G) @ v(Phi[n])).item() for n in range(N)])
            y_cf += epsilon_cf
            t = (1 - w) * (y_cf - y) + w * (y - y_cf)

        if return_dataframe:
            return self.convert_to_pandas(X, w, z, y, y_cf, t)
        else:
            if include_counter_factual:
                return X, w, y, y_cf, t
            else:
                return X, w, y

        # convert to pandas.DataFrame
        data = np.hstack((X, v(w), z, v(y)))
        if include_counter_factual:
            data = np.hstack((data, v(y_cf), v(t)))
        columns = []
        columns_int = []
        for i in range(self.p):
            columns.append('x' + str(i + 1))
        columns.append('w')
        columns_int.append('w')
        for i in range(self.k):
            columns.append('z' + str(i + 1))
            columns_int.append('z' + str(i + 1))
        columns.append('y')
        if include_counter_factual:
            columns.append('y_cf')
            columns.append('t')
        df = pd.DataFrame(data=data, columns=columns)
        df[columns_int] = df[columns_int].astype('int')
        return df

    def convert_to_pandas(self, X, w, z, y, y_cf, t):
        data = np.hstack((X, v(w), z, v(y)))
        if y_cf is not None and t is not None:
            data = np.hstack((data, v(y_cf), v(t)))
        columns = []
        columns_int = []
        for i in range(self.p):
            columns.append('x' + str(i + 1))
        columns.append('w')
        columns_int.append('w')
        for i in range(self.k):
            columns.append('z' + str(i + 1))
            columns_int.append('z' + str(i + 1))
        columns.append('y')
        if y_cf is not None and t is not None:
            columns.append('y_cf')
            columns.append('t')
        df = pd.DataFrame(data=data, columns=columns)
        df[columns_int] = df[columns_int].astype('int')
        return df

    def _set_data(self, X, w, y):
        self.N = len(X)
        self.X_0 = X[w == 0].copy()
        self.X_1 = X[w == 1].copy()
        self.X = np.block([[self.X_0], [self.X_1]])
        self.Phi_0 = self.phi_func(X[w == 0].copy())
        self.Phi_1 = self.phi_func(X[w == 1].copy())
        self.Phi = np.block([[self.Phi_0], [self.Phi_1]])
        self.y_0 = y[w == 0].copy()
        self.y_1 = y[w == 1].copy()
        self.y = np.concatenate([self.y_0, self.y_1])
        self.w = np.sort(w)
        self.w0_end = len(self.X_0)

    def fit(self, X, w, y, **kwargs):
        self._set_data(X, w, y)
        return self._fit(**kwargs)

    def fit_debug(self, X, w, y, **kwargs):
        self._set_data(X, w, y)
        return self._fit_debug(**kwargs)

    def predict(self, X, l, w=None):
        Phi = self.phi_func(X)
        if w == 0:
            y = Phi @ v(self.B[l])
        elif w == 1:
            y = Phi @ v(self.B[l] + self.G[l])
        else:
            y = Phi @ v(self.G[l])
        return y.flatten()

    def plot(self, ax, N=50, showMean=False):
        if self.p != 1:
            raise "Dimension Error"

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        df = self.create_toy_data(N, return_dataframe=True)
        for l in range(self.k):
            z_key = 'z' + str(l + 1)
            df_tmp = df[(df['w'] == 0) & (df[z_key] == 1)]
            x_tmp = df_tmp['x1'].to_numpy()
            y_tmp = df_tmp['y'].to_numpy()
            ax.scatter(x_tmp, y_tmp, color=colors[l], marker='.')

            df_tmp = df[(df['w'] == 1) & (df['z' + str(l + 1)] == 1)]
            x_tmp = df_tmp['x1'].to_numpy()
            y_tmp = df_tmp['y'].to_numpy()
            ax.scatter(x_tmp, y_tmp, color=colors[l], marker='x')

        if showMean:
            x_min = df['x1'].min()
            x_max = df['x1'].max()
            x_support = np.linspace(x_min, x_max, 100)
            for l in range(self.k):
                y = self.predict(v(x_support), l=l, w=0)
                ax.plot(x_support, y, color=colors[l])
                y = self.predict(v(x_support), l=l, w=1)
                ax.plot(x_support, y, linestyle='dashed', color=colors[l])
        return ax


class VBLCMM(BaseLCMM):
    def __init__(self, hyperparams, p, phi_func):
        super().__init__(k=hyperparams['k'], p=p, phi_func=phi_func, sigma2=hyperparams['sigma2'])
        self.lam2_0 = hyperparams['lam2_0']
        self.alpha_0 = hyperparams['alpha_0']
        self.mu_0 = hyperparams['mu_0']
        self.tau_0 = hyperparams['tau_0']
        self.W_0 = hyperparams['W_0']
        self.nu_0 = hyperparams['nu_0']
        # Fixed values
        self.k0 = self.k - 1
        self.ln_C_alpha_0 = ln_C(self.alpha_0)
        self.ln_B_W_0_nu_0 = ln_B(self.W_0, self.nu_0)

    def reset_params_random(self):
        self.pi = np.random.dirichlet(self.alpha_0)
        self.L = np.array([wishart.rvs(scale=self.W_0, df=self.nu_0) for l in range(self.k)])
        if self.p > 1:
            self.M = np.array([np.random.multivariate_normal(self.mu_0, LA.inv(self.tau_0 * self.L[l])) for l in range(self.k)])
        else:
            self.M = np.array([np.random.normal(self.mu_0, 1 / (self.tau_0 * self.L[l])) for l in range(self.k)])
        self.B = np.array([np.random.multivariate_normal(np.zeros(self.m), self.lam2_0 * np.eye(self.m)) for l in range(self.k)])
        self.G = np.array([np.random.multivariate_normal(np.zeros(self.m), self.lam2_0 * np.eye(self.m)) for l in range(self.k)])

    def create_base_instance(self):
        return BaseLCMM(k=self.k, p=self.p, phi_func=self.phi_func, pi=self.pi, M=self.M, L=self.L, B=self.B, G=self.G, sigma2=self.sigma2)

    def predict(self, X, w=None):
        if self.k == 1:
            t_pred = super().predict(X, l=0, w=w)
        else:
            N_new = len(X)
            rho_new = np.empty((N_new, self.k))
            E_x_new_m_L = np.empty((N_new, self.k))
            for n in range(N_new):
                for l in range(self.k):
                    E_x_new_m_L[n,l] = self.p / self.tau_[l] + self.nu_[l] * ( h(X[n] - self.mu_[l]) @ self.W_[l] @ v(X[n] - self.mu_[l]) )
                    rho_new[n,l] = np.exp(
                        self.E_ln_pi[l]
                        + 0.5 * self.E_ln_L[l]
                        - 0.5 * self.p * ln2pi
                        - 0.5 * E_x_new_m_L[n,l]
                    )
            r_new = rho_new / np.sum(rho_new, axis=1)[:, np.newaxis]
            t_pred = np.zeros(N_new)
            for l in range(self.k):
                t_pred += r_new.T[l] * super().predict(X, l=l, w=w)
        return t_pred

    def score(self, X_test, t_test):
        t_pred = self.predict(X_test)
        return mean_squared_error(t_pred, t_test)

    def _fit(self, max_iter=100):
        self._init_random()
        if self.k == 1:
            self._update_all()
            self._calc_VLB()
        else:
            self._update_all()
            for iter in range(max_iter):
                self._update_B()
                self._update_pi()
                self._update_M_L()
                self._update_z_N()
                self._calc_VLB()
                if self.VLB_ == self.VLB_old:
                    self.convergence = iter
                    break
            self.VLB_ += lgamma(self.k+1)
        return self

    def _fit_debug(self, max_iter=100):
        self._init_random()
        if self.k == 1:
            self._update_all()
            self._calc_VLB()
        else:
            show_LB = True
            self._update_all()
            self._calc_VLB()
            for iter in range(max_iter):
                self._update_B()
                self._calc_VLB()
                if self.VLB_ < self.VLB_old and show_LB:
                    print(iter, 'B fix!!')
                    print(self.VLB_old)
                    print(self.VLB_)

                self._update_pi()
                self._calc_VLB()
                if self.VLB_ < self.VLB_old and show_LB:
                    print(iter, 'pi fix!!')
                    print(self.VLB_old)
                    print(self.VLB_)

                self._update_M_L()
                self._calc_VLB()
                if self.VLB_ < self.VLB_old and show_LB:
                    print(iter, 'ML fix!!')
                    print(self.VLB_old)
                    print(self.VLB_)

                self._update_z_N()
                self._calc_VLB()
                if self.VLB_ < self.VLB_old and show_LB:
                    print(iter, 'z fix!!')
                    print(self.VLB_old)
                    print(self.VLB_)

                print(iter, self.VLB_)
                if self.VLB_ == self.VLB_old:
                    self.convergence = iter
                    break

    def _init_random(self):
        # only need to initialize q(z^N)
        self.r_N_ = np.random.dirichlet(np.ones(self.k)*1, size=self.N)
        self._calc_stats_z_N()
        self._weight_by_q_z_N()

        self.r_N_1_ = np.empty(self.k)
        self.B = np.empty((self.k, self.m))
        self.G = np.empty((self.k, self.m))
        self.Sigma_0 = np.empty((self.k, self.m, self.m))
        self.Sigma_1 = np.empty((self.k, self.m, self.m))
        self.Sigma_ = np.empty((self.k, 2*self.m, 2*self.m))
        self.Lam_ = np.empty((self.k, 2*self.m, 2*self.m))
        self.alpha_ = np.empty(self.k)
        self.mu_ = np.empty((self.k, self.p))
        self.tau_ = np.empty(self.k)
        self.W_ = np.empty((self.k, self.p, self.p))
        self.nu_ = np.empty(self.k)

        # Variational lower bound
        self.VLB_ = -np.inf

    def _update_all(self):
        self._update_B()
        self._update_pi()
        self._update_M_L()
        self._update_z_N()

    def _expect_ln_p_y(self):
        self.E_ln_p_y = -0.5 * ( np.sum(self.r_N_) * (ln2pi + np.log(self.sigma2) ) + np.sum( self.r_N_ * self.E_y_bx2) / self.sigma2 )

    def _expect_ln_p_x(self):
        self.E_ln_p_x = 0.5 * ( np.sum(self.N_l * self.E_ln_L) - np.sum(self.N_l) * self.p * ln2pi - np.sum(self.r_N_ * self.E_x_m_L) )

    def _update_B(self):
        for l in range(self.k):
            Lam_0 = (1 / self.sigma2 ) * self.Sigma_prime[l] + (1 / self.lam2_0) * np.eye(self.m)
            Lam_1 = (1 / self.sigma2 ) * self.Sigma_1_prime[l] + (1 / self.lam2_0) * np.eye(self.m)
            Lam_01 = (1 / self.sigma2 ) * self.Sigma_1_prime[l]
            self.Lam_[l] = np.block([[Lam_0, Lam_01], [Lam_01, Lam_1]])
            self.Sigma_[l] = LA.inv(self.Lam_[l]).copy()
            self.Sigma_0[l] = self.Sigma_[l][:self.m,:self.m].copy()
            self.Sigma_1[l] = self.Sigma_[l][self.m:,self.m:].copy()
            m = (1 / self.sigma2) * self.Sigma_[l] @ np.block([[self.Phi_prime[l].T @ v(self.y)], [self.Phi_1_prime[l].T @ v(self.y_1)]])
            b, bw = np.split(m, 2)
            self.B[l] = b.flatten()
            self.G[l] = bw.flatten()

        # Expected values that depend on q(B, Bw)
        self._expect_y_bx2()
        self._expect_bb()
        self._expect_ln_p_B()
        self._expect_ln_p_Bw()
        self._expect_ln_q_BBw()
        self._expect_ln_p_y()

    def _expect_y_bx2(self):
        self.E_y_bx2 = np.empty((self.N, self.k))
        for n in range(self.N):
            A = np.block([np.eye(self.m), self.w[n] * np.eye(self.m)])
            for l in range(self.k):
                b = self.B[l] + self.w[n] * self.G[l]
                self.E_y_bx2[n,l] = self.y[n]**2 \
                            - 2 * self.y[n] * np.sum( b * self.Phi[n] ) \
                            + h(self.Phi[n]) @ ( A @ self.Sigma_[l] @ A.T + v(b) @ h(b) ) @ v(self.Phi[n])

    def _expect_bb(self):
        self.E_bbT = np.empty((self.k, self.m, self.m))
        self.E_bTb = np.empty(self.k)
        for l in range(self.k):
            self.E_bbT[l] = v(self.B[l]) @ h(self.B[l]) + self.Sigma_0[l]
            self.E_bTb[l] = np.trace(self.E_bbT[l])

        self.E_bwbwT = np.empty((self.k, self.m, self.m))
        self.E_bwTbw = np.empty(self.k)
        for l in range(self.k):
            self.E_bwbwT[l] = v(self.G[l]) @ h(self.G[l]) + self.Sigma_1[l]
            self.E_bwTbw[l] = np.trace(self.E_bwbwT[l])

    def _expect_ln_p_B(self):
        self.E_ln_p_B = -0.5 * ( self.k * self.m * (ln2pi + np.log(self.lam2_0)) + np.sum( self.E_bTb / self.lam2_0 ) )

    def _expect_ln_p_Bw(self):
        self.E_ln_p_Bw = -0.5 * ( self.k * self.m * (ln2pi + np.log(self.lam2_0)) + np.sum( self.E_bwTbw / self.lam2_0 ) )

    def _expect_ln_q_BBw(self):
        self.E_ln_q_BBw = 0.5 * ( np.sum( LA.slogdet(self.Lam_)[1]) - self.k * self.m * ( 1 + ln2pi) )

    def _update_z_N(self):
        rho = np.empty((self.N, self.k))
        for n in range(self.N):
            for l in range(self.k):
                rho[n,l] = np.exp(
                                self.E_ln_pi[l] + 0.5 * self.E_ln_L[l] - 0.5 * self.p * (np.log(2) + np.log(np.pi)) - 0.5 * self.E_x_m_L[n,l]
                                - 0.5 * ( ln2pi + np.log(self.sigma2) )
                                - 0.5 / self.sigma2 * self.E_y_bx2[n,l]
                            )
                if rho[n,l]==0:
                    rho[n,l] = 1e-300
        self.r_N_ = rho / np.sum(rho, axis=1)[:,np.newaxis]

        # Expected values that depend on q(z^N)
        self._calc_stats_z_N()
        self._weight_by_q_z_N()
        self._expect_ln_p_z_N()
        self._expect_ln_q_z_N()
        self._expect_ln_p_y()
        self._expect_ln_p_x()

    def _calc_stats_z_N(self):
        self.N_l = np.sum(self.r_N_, axis=0)
        self.x_bar = np.empty((self.k, self.p))
        self.S = np.zeros((self.k, self.p, self.p))
        for l in range(self.k):
            self.x_bar[l] = np.sum(self.X[:,:] * self.r_N_.T[l][:,np.newaxis], axis=0) / self.N_l[l]
            """
            self.x_bar[l] = np.zeros(self.p)
            for n in range(self.N):
                self.x_bar[l] += self.r_N_[n,l] * self.X[n,:]
            self.x_bar[l] /= self.N_l[l]
            """
            # X_x_bar = self.X[:,1:] - self.x_bar[l]
            # self.S[l] = ( (X_x_bar * self.r_N_.T[l][:,np.newaxis]).T @ X_x_bar) / self.N_l[l]
            for n in range(self.N):
                self.S[l] += self.r_N_[n,l] * ( v(self.X[n][:] - self.x_bar[l]) @ h(self.X[n][:] - self.x_bar[l]) )
            self.S[l] /= self.N_l[l]

    def _weight_by_q_z_N(self):
        self.Phi_0_prime = np.empty((self.k, self.w0_end, self.m))
        self.Phi_1_prime = np.empty((self.k, self.N - self.w0_end, self.m))
        self.Phi_prime = np.empty((self.k, self.N, self.m))
        self.Sigma_0_prime = np.empty((self.k, self.m, self.m))
        self.Sigma_1_prime = np.empty((self.k, self.m, self.m))
        self.Sigma_prime = np.empty((self.k, self.m, self.m))
        for l in range(self.k):
            self.Phi_prime[l] = self.Phi * self.r_N_[:,l,np.newaxis]
            self.Phi_0_prime[l] = self.Phi_prime[l][:self.w0_end]
            self.Phi_1_prime[l] = self.Phi_prime[l][self.w0_end:]
            self.Sigma_0_prime[l] = self.Phi_0_prime[l].T @ self.Phi_0
            self.Sigma_1_prime[l] = self.Phi_1_prime[l].T @ self.Phi_1
            self.Sigma_prime[l] = self.Sigma_0_prime[l] + self.Sigma_1_prime[l]

    def _expect_ln_p_z_N(self):
        self.E_ln_p_z_N = np.sum(self.r_N_ * self.E_ln_pi)

    def _expect_ln_q_z_N(self):
        self.r_N_[np.round(self.r_N_, decimals=10) == 0] += sys.float_info.min
        self.E_ln_q_z_N = np.sum(self.r_N_ * np.log(self.r_N_))

    def _update_pi(self):
        self.alpha_ = self.alpha_0 + self.N_l

        # Expected values that depend on q(pi)
        self._expect_ln_pi()
        self._expect_ln_p_pi()
        self._expect_ln_q_pi()
        self._expect_ln_p_z_N()

    def _expect_ln_pi(self):
        self.E_ln_pi = psi(self.alpha_) - psi(np.sum(self.alpha_))

    def _expect_ln_p_pi(self):
        self.E_ln_p_pi = np.sum( (self.alpha_0 - 1) * self.E_ln_pi ) + self.ln_C_alpha_0

    def _expect_ln_q_pi(self):
        self.E_ln_q_pi = np.sum( (self.alpha_ - 1) * self.E_ln_pi) + ln_C(self.alpha_)

    def _update_M_L(self):
        self.tau_ = self.tau_0 + self.N_l
        self.mu_ = ( self.tau_0 * self.mu_0[np.newaxis] + self.x_bar * self.N_l[:,np.newaxis] ) / self.tau_[:,np.newaxis]
        """
        t = np.zeros_like(self.mu_)
        for l in range(self.k):
            t[l] = ( self.tau_0 * self.mu_0 + self.x_bar[l] * self.N_l[l] ) / self.tau_[l]
        print( self.mu_ - t )
        """
        self.W_ = LA.inv(
                    LA.inv(self.W_0)
                  + self.N_l[:,np.newaxis,np.newaxis] * self.S
                  + (self.tau_0 * self.N_l  / (self.tau_0 + self.N_l))[:,np.newaxis,np.newaxis] * np.array([v(x) @ h(x) for x in (self.x_bar - self.mu_0[np.newaxis])])
                )
        """
        t = np.zeros_like(self.W_)
        for l in range(self.k):
            t[l] = LA.inv(
                LA.inv(self.W_0)
                + self.N_l[l] * self.S[l]
                + self.tau_0 * self.N_l[l] / (self.tau_0 + self.N_l[l]) * ( v(self.x_bar[l] - self.mu_0) @ h(self.x_bar[l] - self.mu_0) )
            )
        print(self.W_ - t)
        """
        self.nu_ = self.nu_0 + self.N_l

        # Expected values that depend on q(M, L)
        self._expect_M_L()
        self._expect_ln_p_x()
        self._expect_ln_p_M_L()
        self._expect_ln_q_M_L()

    def _expect_M_L(self):
        self.E_ln_L = np.empty(self.k)
        for l in range(self.k):
            self.E_ln_L[l] = np.sum( np.array([psi( 0.5 * (self.nu_[l] - d) ) for d in range(self.p)]) ) + self.p * np.log(2) + LA.slogdet(self.W_[l])[1]

        self.E_x_m_L = np.empty((self.N, self.k))
        for n in range(self.N):
            for l in range(self.k):
                self.E_x_m_L[n,l] = self.p / self.tau_[l] + self.nu_[l] * ( h(self.X[n,:] - self.mu_[l]) @ self.W_[l] @ v(self.X[n,:] - self.mu_[l]) )

    def _expect_ln_p_M_L(self):
        E_m_mu_L = np.empty(self.k)
        for l in range(self.k):
            E_m_mu_L[l] = self.p / self.tau_[l] + self.nu_[l] * (h(self.mu_[l] - self.mu_0) @ self.W_[l] @ v(self.mu_[l] - self.mu_0))

        self.E_ln_p_M_L = 0.5 * np.sum(
                            self.p * ( np.log(self.tau_0) - ln2pi )
                            + self.E_ln_L
                            - self.tau_0 * E_m_mu_L
                        ) + (
                            + self.k * self.ln_B_W_0_nu_0
                            + 0.5 * (self.nu_0 - self.p - 1) * np.sum(self.E_ln_L)
                            - 0.5 * np.sum( self.nu_ * np.trace( LA.inv(self.W_0) @ self.W_, axis1=1, axis2=2) )
                        )

    def _expect_ln_q_M_L(self):
        H = np.empty(self.k)
        for l in range(self.k):
            H[l] = - ln_B(self.W_[l], self.nu_[l]) - 0.5 * (self.nu_[l] - self.p - 1) * self.E_ln_L[l] + 0.5 * self.nu_[l] * self.p
        self.E_ln_q_M_L = np.sum( 0.5 * self.E_ln_L + 0.5 * self.p * (np.log(self.tau_) - ln2pi ) - 0.5 * self.p - H )

    def _calc_VLB(self, epsilon=5):
        self.VLB_old = self.VLB_
        self.VLB_ = (self.E_ln_p_y + self.E_ln_p_x - self.N * np.log(2) + self.E_ln_p_z_N + self.E_ln_p_pi + self.E_ln_p_M_L + self.E_ln_p_B + self.E_ln_p_Bw) \
                    - (self.E_ln_q_z_N + self.E_ln_q_pi + self.E_ln_q_M_L + self.E_ln_q_BBw)
        self.VLB_ = round(self.VLB_, epsilon)


class RandomInitVBLCMM():

    def __init__(self, estimator):
        self.estimator = estimator
        self.estimator_list = []
        self.best_estimator_ = None

    def fit(self, X, w, y, init_num, max_iter=100):
        N = len(X)
        if self.estimator.k == 1:
            print('N=%d, fit k=1' % N)
            self.estimator.fit(X, w, y, max_iter=max_iter)
            self.estimator_list.append(copy.deepcopy(self.estimator))
            self.best_estimator_ = self.estimator_list[0]
        else:
            print('init', end='')
            for i in range(init_num):
                print("\rN=%d, fit k=%d: VB random init %d/%d" % (N, self.estimator.k, i+1, init_num), end='')
                self.estimator.fit(X, w, y, max_iter=max_iter)
                self.estimator_list.append( copy.deepcopy(self.estimator) )
            print("\rN=%d, fit k=%d: VB random init %d/%d" % (N, self.estimator.k, init_num, init_num))
            self.best_estimator_ = max(self.estimator_list, key=lambda x: x.VLB_)

    def show_VLBs(self, ax):
        vlbs = np.array([ m.VLB_ for m in self.estimator_list])
        x = np.linspace(self.estimator.k - 0.1, self.estimator.k + 0.1, len(vlbs))
        ax.scatter(x, vlbs, marker='x')


class MetaVBLCMM():

    def __init__(self, hyperparams_list, p, phi_func):
        self.hyperparams_list = hyperparams_list
        self.meta_model_list = []
        self.best_estimator_list = []
        self.q_k = np.zeros(len(self.hyperparams_list))
        self.p = p
        self.phi_func = phi_func

    def fit(self, X, w, y, init_num=30, max_iter=100):
        for l, hyperparams in enumerate(self.hyperparams_list):
            # print('hparam', l)
            model_l = RandomInitVBLCMM(VBLCMM(hyperparams, self.p, self.phi_func))
            model_l.fit(X, w, y, init_num=init_num, max_iter=max_iter)
            self.meta_model_list.append(model_l)
            self.best_estimator_list.append( model_l.best_estimator_ )

        C = np.sum([ np.exp(m.VLB_ + np.log(np.math.factorial(m.k))) for m in self.best_estimator_list])
        if C > 0:
            self.q_k = np.array([ np.exp(m.VLB_ + np.log(np.math.factorial(m.k))) for m in self.best_estimator_list]) / C
        else:
            max_index = np.argmax([ m.VLB_ + np.log(np.math.factorial(m.k)) for m in self.best_estimator_list])
            self.q_k[max_index] = 1.0

    def predict(self, X, w=None):
        y = 0
        for l, q in enumerate(self.q_k):
            if q != 0:
                y += q * self.best_estimator_list[l].predict(X, w=w)
        return y

    def score(self, X_test, t_test):
        t_pred = self.predict(X_test)
        return mean_squared_error(t_pred, t_test)

    def score_by_model(self, X_test, t_test):
        score_by_model = []
        for best_estimator in self.best_estimator_list:
            score_by_model.append(best_estimator.score(X_test, t_test))
        return score_by_model

    def show_VLBs(self, ax):
        for m in self.meta_model_list:
            m.show_VLBs(ax)
