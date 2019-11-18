# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import pickle
import glob
import os
import pdb

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
import numpy as np
import itertools
import scipy
import scipy.stats
from random import shuffle
from label_lines import *

#mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = '6'
linewidth = '0.3'
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.linewidth'] = linewidth
mpl.rcParams['xtick.major.width'] = linewidth
mpl.rcParams['ytick.major.width'] = linewidth
label_fontsize = 6

linestyles = ['-', '--', '-.', '-', '-', '--', '-.', '-']

colors = ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
          "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

def plot_averaged(plot_name, plot_entries, xvals, yrange=None):
    run_dir = "runs"
    plot_dir = "plots"
    loss_key = "test_errors"
    ylabel = "Test error (%)"

    # Positions of the labels in x axis range, i.e. epoch number
    #xvals

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    max_seen = 0
    plt.cla()
    fig = plt.figure(figsize=(3.3,2))
    ax = fig.add_subplot(111)

    ax.set_prop_cycle("color", colors)
    #ax.set_prop_cycle("linestyle", linestyles)
    line_idx = 0

    for plot_entry in plot_entries:
        fname_grob = plot_entry["fname"]
        data_files = glob.glob(fname_grob)

        if len(data_files) == 0:
            raise Exception("No files found matching path: {}".format(fname_grob))

        errors_lists = []
        for fname in data_files:
            print("(ALL) processing run ", fname)
            with open(fname, 'rb') as fdata:
                rd = pickle.load(fdata)

                values = rd[loss_key]
                # convert to errors
                errors = [100.0 - val for val in values]
                #pdb.set_trace()
                #print("losses: {}".format(losses))
                print("Final test error {} for {}".format(errors[-1], plot_entry["label"]))

                # Remove outlier runs
                if errors[-1] < 20.0:
                    errors_lists.append(errors.copy())

                    max_test_loss = max(errors)
                    if max_test_loss > max_seen:
                        max_seen = max_test_loss
                    max_epoch = len(errors)

        ## Aggregate and plots
        n = len(errors_lists)
        errors_avg = [0.0 for i in range(len(errors_lists[0]))]
        errors_low = [0.0 for i in range(len(errors_lists[0]))]
        errors_hi = [0.0 for i in range(len(errors_lists[0]))]
        #pdb.set_trace()

        # Apply a smoothing filter
        box_pts = 10
        box = np.ones(box_pts)/box_pts
        for i in range(len(errors_lists)):
            errors_lists[i] = np.convolve(errors_lists[i], box, mode='valid')

        # Change from a list of runs to a list of epochs
        errors = np.array(errors_lists).T.tolist()

        for i in range(len(errors)):
            sem = scipy.stats.sem(errors[i])
            errors_avg[i] = np.mean(errors[i])
            errors_low[i] = errors_avg[i] - sem
            errors_hi[i] = errors_avg[i] + sem
        
        x = range(len(errors_avg))
        ax.plot(
            x,
            errors_avg,
            label=plot_entry["label"],
            linestyle=linestyles[line_idx]) #linestyle=next(linestyles)
        ax.fill_between(x, errors_low, errors_hi, alpha=0.3)

        line_idx += 1
        print("Average final test error {} for {}".format(errors_avg[-1], plot_entry["label"]))

    print("Finalizing plot")
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.xlim([0, max_epoch-box_pts])
    pdb.set_trace()

    if yrange is not None:
        plt.ylim(yrange)
    else:
        plt.ylim([0, max_seen])

    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    ax.grid(False)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in', right="on")
    labelLines(plt.gca().get_lines(), align=False, fontsize=label_fontsize, xvals=xvals)
    #ax.legend(fontsize=5, handlelength=8, loc='center left', bbox_to_anchor=(1, 0.5))
    figname = "{}/{}.pdf".format(plot_dir, plot_name)

    fig.savefig(figname, bbox_inches='tight', pad_inches=0)
    print("saved", figname)

plot_averaged(
    plot_name="test_resnet110_V2",
    plot_entries = [
    {"fname": "runs/cifar10/*resnet110*scsg*.pkl", "label": "SCSG"},
    {"fname": "runs/cifar10/*resnet110*sgd*.pkl", "label": "SGD"},
    {"fname": "runs/cifar10/*resnet110*svrg*.pkl", "label": "SVRG"},
    ],
    xvals=[175, 210, 100],
    yrange=[6.0, 40.0])
