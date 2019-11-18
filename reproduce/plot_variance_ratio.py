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
import matplotlib.ticker as plticker

import numpy as np
import itertools
import random
from random import shuffle
import pdb

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
import numpy as np
import itertools
from random import shuffle
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

colors = ["black", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
          "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

def plot_variance_ratios(plot_name, data_files_grob, xvals):
    ylabel = "SVR Variance / SGD Variance"

    keys_to_show = ['2%', '11%', '33%', '100%']
     # Position of in-plot labels along the x axis

    epochs = []
    ratios = []
    vr_variances = []
    gradient_variances = []

    trace_data = {}

    data_files = glob.glob(data_files_grob)

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]

    data_files.sort(key=natural_keys)

    #pdb.set_trace()

    for fname in data_files:
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

                ratio_points = (np.array(rd["vr_step_variances"])/np.array(rd["gradient_variances"])).tolist()

                for i, ep in enumerate(eval_points):
                    ep_name = "{0:.0f}%".format(100*ep)
                    if ep_name not in trace_data.keys():
                        trace_data[ep_name] = [ratio_points[i]]
                    else:
                        trace_data[ep_name].append(ratio_points[i])

    plt.cla()
    fig = plt.figure(figsize=(3.2,2))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle("color", colors)

    #pdb.set_trace()
    for ep_name, data in trace_data.items():
        if ep_name in keys_to_show:
            ax.plot(epochs, data, ".",
                label=ep_name) #, linestyle=next(linestyles))
        if ep_name == "100%":
            print("100p epochs:", epochs)
            print("ratios: ", data)


    print("Finalizing plot")
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    ax.set_yscale("log", basey=2)
    ax.set_yticks([2**(-i) for i in range(0, 11)])
    plt.ylim([1e-3, 3])
    plt.xlim([0.0, 240])

    # Horizontal line at 1
    #plt.axhline(y=1.0, color="#000000", linestyle='--')
    #plt.axhline(y=2.0, color="#000000", linestyle='--')
    ax.axhspan(1, 2, alpha=0.3, facecolor='red', edgecolor=None)
    # Step size reduction indicators
    plt.axvline(x=150.0, color="brown", linestyle='--')
    plt.axvline(x=220.0, color="brown", linestyle='--')


    #loc = plticker.LogLocator(base=2.0)
    #ax.yaxis.set_major_locator(loc)
    #plt.tick_params(axis='y', which='minor')

    ax.grid(False)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in', right="on")
    labelLines(plt.gca().get_lines(), align=False, fontsize=label_fontsize, xvals=xvals)
    figname = "{}/{}.pdf".format(plot_dir, plot_name)

    fig.savefig(figname, bbox_inches='tight', pad_inches=0)
    print("saved", figname)

################## xvals are low percentages to high
plot_variance_ratios(plot_name = "variance_ratios_densenet",
    data_files_grob = "data/variance1/var-*.pkl", xvals = [200, 200, 200, 200])
plot_variance_ratios(plot_name = "variance_ratios_lenet",
    data_files_grob = "data/variance-lenet/*.pkl",  xvals = [210, 200, 190, 210])
plot_variance_ratios(plot_name = "variance_ratios_small-resnet",
    data_files_grob = "data/variance-small-resnet/*.pkl",  xvals = [180, 180, 180, 210])
plot_variance_ratios(plot_name = "variance_ratios_resnet110",
    data_files_grob = "data/variance-resnet110/*.pkl",  xvals = [180, 180, 180, 210])


# Soft versions
if True:
    plot_variance_ratios(plot_name = "soft_variance_ratios_densenet",
        data_files_grob = "data/variance-soft/*densenet*.pkl", xvals = [200, 200, 200, 200])
    plot_variance_ratios(plot_name = "soft_variance_ratios_lenet",
        data_files_grob = "data/variance-soft/*default*.pkl",  xvals = [210, 200, 190, 210])
    plot_variance_ratios(plot_name = "soft_variance_ratios_small-resnet",
        data_files_grob = "data/variance-soft/*resnet*.pkl",  xvals = [180, 180, 180, 210])
