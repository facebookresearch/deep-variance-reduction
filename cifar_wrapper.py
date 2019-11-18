# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import pdb

class CIFAR10_Wrapper(torch.utils.data.Dataset):

    def __init__(self, root, train, download, transform):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train,
            download=download, transform=transform)

        self.transformed_cache = {}
        self.access_total = 0
        self.cache_hit = 0
        self.access_since_retransform = 0

    def __getitem__(self, index):
        #print(self.transformed_cache.keys())
        if index in self.transformed_cache.keys():
            item = self.transformed_cache[index]
            self.cache_hit += 1
            #print("Using cache: ", index)
        else:
            item = self.dataset[index]
            self.transformed_cache[index] = item
            #pdb.set_trace()
            #print("Writing cache: ", index)

        self.access_total += 1
        self.access_since_retransform += 1
        #print("since retransform: ", self.access_since_retransform)
        #print("total: ", self.access_total)
        return item

    def __len__(self):
        return len(self.dataset)
        #return 128

    # flushes the cache of transformed images
    def retransform(self):
        print("total calls retransform: {}, cache hits: {}".format(
            self.access_since_retransform, self.cache_hit))
        #print("total: ", self.access_total)
        self.transformed_cache = {}
        self.access_since_retransform = 0
        self.cache_hit = 0
