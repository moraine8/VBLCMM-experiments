# coding: utf-8
import datetime
import csv
import os
import time
import json
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean

from VBLCMM import *
from basis_function import *


def make_output_directory(base_dir):
    datetime_str = datetime.datetime.today().strftime("%Y%m%d%H%M%S")  # for output file name
    output_directory = base_dir + datetime_str + '/'
    os.makedirs(output_directory)
    return output_directory


def make_figure(results_dir):
    with open(results_dir + "condition.json", "r") as f:
        condition = json.load(f)

    N_list = condition['N_train_list']

    results_means = []
    results_stds = []
    files = sorted(glob.glob(results_dir + "*.csv"))
    for file in files:
        results = np.loadtxt(file, delimiter=',')
        results_means.append(np.mean(results, axis=0))
        results_stds.append(np.std(results, axis=0))
    label_dict = {
        '0_Proposal': 'Proposal',
        '1_DoubleModel': '(i)',
        '2_JamesStein': '(ii)',
        '3_SMSE': '(iii)'
    }
    fig = plt.figure(figsize=[6.4,3.8])
    ax = fig.add_subplot()
    for (file, results_mean, results_std) in zip(files, results_means, results_stds):
        filename = os.path.splitext(os.path.basename(file))[0]
        if filename in label_dict:
            label = label_dict[filename]
        else:
            label = filename
        ax.plot(N_list, results_mean, marker='.', label=label)
        #ax.errorbar(x, results_mean, yerr=results_std, label=filename)
    ax.set_xlabel("Number of training data (N)")
    ax.set_ylabel("MSEs of the prediction")
    ax.legend()
    plt.savefig(results_dir + "graph.pdf")
    print('saved figure at %s' % results_dir)

if __name__ == '__main__':
    make_figure('./results/experiment3/exp1000/')
