# coding: utf-8
import numpy as np
import pandas as pd

def h(vec):
    # 1-dim p vector to p x 1 matrix
    return vec.reshape(1, -1)
def v(vec):
    # 1-dim p vector to 1 x p matrix
    return vec.reshape(-1, 1)
