# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pdb
import os

class VRSamplerIter(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.i = 0

    def __next__(self):
        #if self.sampler.creation_process != os.getpid():
        #    print("__next__ called on child process")

        self.i += 1
        if self.i > self.sampler.nbatches:
            raise StopIteration
        else:
            return self.sampler.batches[self.i-1]

    def __len__(self):
        return self.sampler.nbatches

class VRSampler(object):
    """Wraps two samplers to craete a sampler object suitable for use with
    variance reduction. methods

    Args:
        initial_sampler (Sampler): Base sampler for initial ordering.
        order (string) Either inorder, perm or random.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        list(VRSampler(range(10), order="inorder", batch_size=3, drop_last=False))
    """

    def __init__(self, order, batch_size, dataset_size, drop_last=False):
        self.order = order
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.creation_process = os.getpid()

        self.reorder()

    def reorder(self):
        if self.creation_process != os.getpid():
            raise Exception("reorder called on child process, which is bad. {} got: {}".format(self.creation_process, os.getpid()))

        print("Reordering instances: {}".format(self.order))
        if self.order == "perm":
            idx_list = torch.randperm(self.dataset_size)
        else:
            idx_list = (torch.rand(self.dataset_size)*self.dataset_size).long()

        # Generate initial minibatches
        self.batches = []
        batch = []
        for idx in idx_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                self.batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            self.batches.append(batch)

        self.nbatches = len(self.batches)
        #pdb.set_trace()

    def __iter__(self):
        print("Sampler __iter__")
        return VRSamplerIter(self)

    def __len__(self):
        return self.nbatches
