# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import pickle
import glob
import os
import re

import numpy as np
import itertools
import random
from random import shuffle
import pdb

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
import numpy as np
import itertools
from random import shuffle
from matplotlib.ticker import FuncFormatter
from label_lines import *

run_dir = "runs"
plot_dir = "plots"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = '6'
linewidth = '0.3'
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.linewidth'] = linewidth
mpl.rcParams['xtick.major.width'] = linewidth
mpl.rcParams['ytick.major.width'] = linewidth
label_fontsize = 6

linestyles = itertools.cycle(('-', '--', '-.', ':'))

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
          "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

def plot_variance_raw(plot_name, data_files, labels):
    ylabel = "SVRG Variance"

    xvals = [0.7, 0.7] # Position of in-plot labels along the x axis

    epochs = []

    plt.cla()
    fig = plt.figure(figsize=(3.2,2))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle("color", colors)


    for fname, label in zip(data_files, labels):
        print("(ALL) processing file ", fname)
        with open(fname, 'rb') as fdata:
            rd = pickle.load(fdata)
            #pdb.set_trace()
            if 'batch_indices' in rd:
                print("Has batch indices")
                # Calculate x axis for plotting
                batch_indices = np.array(rd["batch_indices"])
                nk = len(batch_indices)
                if max(batch_indices) == min(batch_indices):
                    eval_points = np.array(range(nk))/nk
                else:
                    eval_points = batch_indices/max(batch_indices)

                epochs.append(rd["epoch"])
                #pdb.set_trace()

                var_points = rd["vr_step_variances"]
                #pdb.set_trace()

                ax.plot(eval_points, var_points, label=label)

    # Only compared data from the same epoch
    if len(set(epochs)) > 1:
        print("More than one epoch encountered: {}".format(epochs))


    print("Finalizing plot")
    plt.xlabel('Progress within epoch')
    plt.ylabel(ylabel)
    plt.ylim([0, 0.7])
    plt.xlim([0.0, 1.0])

    # Format x axis as percentage
    def myfunc(x, pos=0):
     return '%1.0f%%'%(100*x)
    ax.xaxis.set_major_formatter(FuncFormatter(myfunc))


    ax.grid(False)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in', right="on")
    labelLines(plt.gca().get_lines(), align=False, fontsize=label_fontsize, xvals=xvals)
    figname = "{}/{}.pdf".format(plot_dir, plot_name)

    fig.savefig(figname, bbox_inches='tight', pad_inches=0)
    print("saved", figname)

##################
plot_variance_raw(plot_name = "variance_transform",
                     data_files = [
        "data/variance-locking/default-lenet-shortvariance_epoch3.pkl",
        "data/variance-locking/default-lenet-short_tlockvariance_epoch3.pkl",
        ],
        labels = ["No locking", "Transform locking"])
