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
    'method': method,
    'seed': seed,
    'problem': 'imagenet',
    'architecture': 'resnet18',
    'momentum': 0.9,
    'lr': 0.1,
    'decay': 0.0001,
    'lr_reduction': "every30",
    'batch_size': 256,
    'epochs': 90,
    'save_model': True,
    'full_checkpointing': True,
    'log_interval': 80,
    }

run.run(runargs)
