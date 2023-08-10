# coding: utf-8
import numpy as np

def linear(X):
    """
    \phi(\bm{x}) = 1 + x_1 + x_2 + ... + x_p
    """
    if X.ndim != 2:
        raise "Basis Function Error"
    return np.insert(X, 0, 1.0, axis=1)

def quadratic(X):
    """
    \phi(\bm{x}) = 1 + x_1 + x_2 + ... + x_p + x_1^2 + x_2^2 + ... + x_p^2
    """
    if X.ndim != 2:
        raise "Basis Function Error"
    XX = np.power(X, 2)
    X_ret = np.append(X, XX, axis=1)
    return np.insert(X_ret, 0, 1.0, axis=1)