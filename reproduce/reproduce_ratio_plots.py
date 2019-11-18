# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import run

archs = ['default', 'resnet-small', 'densenet-40-36', 'resnet110']

try:
    pindex = int(sys.argv[1])
    print(f"problem index {pindex}")
except:
    pindex = 0

arch = archs[pindex]

runargs = {
    'problem': 'cifar10',
    'architecture': arch,
    'method': "svrg",
    'logfname': 'reproduce-variance-ratios-{}'.format(arch),
    'momentum': 0.9,
    'decay': 0.0001,
    'lr': 0.1,
    'lr_reduction': "150-225",
    'batch_size': 128,
    'epochs': 250,
    'log_diagnostics': True,
    }

run.run(runargs)
