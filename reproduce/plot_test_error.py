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
from random import shuffle
from label_lines import *

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
    #plot_name = "test_error1"
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
                errors_lists.append(errors.copy())

                max_test_loss = max(errors)
                if max_test_loss > max_seen:
                    max_seen = max_test_loss
                max_epoch = len(errors)

        ## Aggregate and plots
        n = len(errors_lists)
        errors_avg = [0.0 for i in range(len(errors_lists[0]))]
        for i in range(n):
            for j in range(len(errors_avg)):
                errors_avg[j] += float(errors_lists[i][j]/n)

        #pdb.set_trace()
        ax.plot(
            range(len(errors_avg)),
            errors_avg,
            label=plot_entry["label"],
            linestyle=linestyles[line_idx]) #linestyle=next(linestyles)

        line_idx += 1
        print("Average final test error {} for {}".format(errors_avg[-1], plot_entry["label"]))

    print("Finalizing plot")
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.xlim([0, max_epoch])

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


#######################################################
######################################################


plot_averaged(
    plot_name="test_resnet110_V2",
    plot_entries = [
    {"fname": "runs/cifar10/*resnet110*scsg*.pkl", "label": "SCSG"},
    {"fname": "runs/cifar10/*resnet110*sgd*.pkl", "label": "SGD"},
    {"fname": "runs/cifar10/*resnet110*svrg*.pkl", "label": "SVRG"},
    ],
    xvals=[175, 210, 250],
    yrange=[0, 40.0])

# plot_averaged(
#     plot_name="test_lenet",
#     plot_entries = [
#     {"fname": "data/runs-large/cifar10-default-scsg-m0_9d0_0001lr0_1sl1e-06epochs300bs128pb_Falseis_10drop_Falsebn_Truereduct_150-225seed_*.pkl", "label": "SCSG"},
#     {"fname": "data/runs-large/cifar10-default-sgd-m0_9d0_0001lr0_1sl1e-06epochs300bs128pb_Falseis_10drop_Falsebn_Truereduct_150-225seed_*.pkl", "label": "SGD"},
#     {"fname": "data/runs-large/cifar10-default-svrg-lr0_1-m0_9-d0_0001-epochs300bs128drop_Falsebn_Truevr_from_1bn_recal_Truereduct_150-225seed_*.pkl", "label": "SVRG"},
#     ],
#     xvals=[175, 210, 250],
#     yrange=[20.0, 40.0])

# plot_averaged(
#     plot_name="test_imagenet",
#     plot_entries = [
#     {"fname": "data/imagenet-vr/*scsg*.pkl", "label": "SCSG"},
#     {"fname": "data/imagenet-sgd/*.pkl", "label": "SGD"},
#     {"fname": "data/imagenet-vr/*svrg*.pkl", "label": "SVRG"},
#     ],
#     xvals=[22, 45, 10],
#     yrange=[30.0, 70.0])
