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
import math

import numpy as np
import scipy.stats
import itertools
import random
from random import shuffle
from collections import OrderedDict

#import normalization.positive_normalization
#from normalization import positive_normalization
#from normalization import *
#from label_lines import *

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)

#mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = '6'
linewidth = '0.3'
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['axes.linewidth'] = linewidth
mpl.rcParams['xtick.major.width'] = linewidth
mpl.rcParams['ytick.major.width'] = linewidth
label_fontsize = 3

linestyles = ['-', '--', '-.', '-', '-', '--', '-.', '-']

#colors = ["#659BC9", "#551a8b", "#e41a1c", "#377eb8"]
#101, 155, 201 is 	#659BC9. 236, 163, 57 is #ECA339
colors = ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
          "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

loss_type="test_errors"
ylim=(20, 70) #100-77
xlim=(0, 90)

figname = "finetuning_densenet.pdf"

runs = [
    '/checkpoint/adefazio/opt/vr_densenet_resume/20191022_070415j60d2vd4/job-0/log/run123/current_runinfo.pkl',
    '/checkpoint/adefazio/opt/vr_densenet_sweep2/20191018_113200kfr3mjvy/job-4/log/run123/current_runinfo.pkl',
    #'/checkpoint/adefazio/opt/vr_densenet_sweep2/20191018_113200kfr3mjvy/job-5/log/run123/current_runinfo.pkl',
]

traces = []

plt.cla()
scalefactor = 0.85
fig = plt.figure(figsize=(scalefactor*3.3,scalefactor*2))
ax = fig.add_subplot(111)

ax.set_prop_cycle("color", colors)
plt.xlabel('Epoch')

plt.ylabel("Test error (%)")
legend_position = 'upper right'

idx = 0
for fname in runs:
    print("(ALL) processing run ", fname)
    with open(fname, 'rb') as fdata:
        rd = pickle.load(fdata)
        args = rd['args']
        losses = rd[loss_type]
        losses = losses[:xlim[1]]
        losses = 100.0 - np.array(losses)

        legend = f"from {args.vr_from_epoch:2d} ({min(losses):1.2f}%)"

        if len(losses) > 0:
            x = list(range(1,len(losses)+1))
            y = losses.copy()

            ax.plot(x, y, label=legend,
                linestyle=linestyles[idx % len(linestyles)])
    idx += 1

#pdb.set_trace()

print("Finalizing plot")
if ylim is not None:
    plt.ylim(ylim)
if xlim is not None:
    plt.xlim(xlim)

baseline = 23.22
ax.axhline(baseline, color="r", label=f"Baseline ({baseline:1.2f}%)")

ax.grid(False)
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in', right=True)
#labelLines(plt.gca().get_lines(), align=False, fontsize=label_fontsize, xvals=xvals)
ax.legend(fontsize=5, handlelength=2, loc=legend_position) #bbox_to_anchor=(1, 0.5)

fig.savefig(figname, bbox_inches='tight', pad_inches=0)
print("saved", figname)
