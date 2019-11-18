# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import run

methods = ["sgd", "recompute_svrg", "scsg"]
try:
    pindex = int(sys.argv[1])
    seed = int(sys.argv[2])
    print(f"problem index {pindex}")
except:
    pindex = 0
    seed = 0

method = methods[pindex]

runargs = {
    'problem': 'cifar10',
    'architecture': 'resnet110', #'resnet110',
    'method': method,
    'seed': seed,
    'momentum': 0.9,
    'decay': 0.0001,
    'lr': 0.05,
    'lr_reduction': "150-225",
    'batch_size': 128,
    'epochs': 250,
    'log_diagnostics': False,
    'save_model': True,
    }

run.run(runargs)
