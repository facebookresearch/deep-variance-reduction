# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import run

vr_froms = [1, 21, 41, 61, 81, 1234]

try:
    pindex = int(sys.argv[1])
    print(f"problem index {pindex}")
except:
    pindex = 0
    seed = 0

method = methods[pindex]

runargs = {
    'vr_from_epoch': vr_froms[pindex],
    'method': 'recompute_svrg',
    'problem': 'imagenet',
    'architecture': 'resnet50',
    'momentum': 0.9,
    'lr': 0.1,
    'decay': 0.0001,
    'lr_reduction': "every30",
    'batch_size': 256,
    'epochs': 100,
    'save_model': True,
    'full_checkpointing': True,
    'log_interval': 80,
    }

run.run(runargs)
