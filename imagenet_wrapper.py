# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
import torchvision
import torchvision.transforms as transforms
import caching_transforms
import torch.utils.data
import pdb
import logging
import numpy as np

from multiprocessing.sharedctypes import RawArray
from ctypes import Structure, c_double

class ImagenetWrapper(torch.utils.data.Dataset):

    def __init__(self, root, lock_transforms):
        global transform_instances
        self.dataset = datasets.ImageFolder(root)

        self.nimages = len(self.dataset)
        self.rand_per_image = 6

        self.transform_instances = RawArray(c_double, self.rand_per_image*self.nimages)
        transform_instances = self.transform_instances

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # These modified transforms take the random numbers they need explicitly
        # as arguments to the transform method.
        self.crop = caching_transforms.RandomResizedCrop(224)
        self.flip = caching_transforms.RandomHorizontalFlip()

        # Initial transformation cache created.
        self.retransform()

    # Must be called on each child process
    def child_initialize(self, worker_id):
        global transform_instances

        transform_instances = self.transform_instances # self is parent processes version
        global_worker_id = worker_id
        print("child process: {}".format(global_worker_id))

    def __getitem__(self, index):
        global transform_instances

        item = self.dataset[index]
        img, lbl = item

        if index == 0:
            self.print_rands()

        # Apply transforms using saved random numbers
        start = index*self.rand_per_image
        transformed_img = self.crop.transform(img,
            transform_instances[start], transform_instances[start+1], transform_instances[start+2],
            transform_instances[start+3], transform_instances[start+4])
        transformed_img = self.flip.transform(transformed_img, transform_instances[start+5])

        transformed_img = self.normalize(caching_transforms.to_tensor(transformed_img))

        return (transformed_img, lbl)

    def __len__(self):
        #return 1024
        return len(self.dataset)

    def retransform(self):
        np_instances = np.frombuffer(self.transform_instances)

        # Generate all the random numbers
        logging.info("Generating {} random numbers ...".format(len(self.transform_instances)))
        np_instances[:] = np.random.uniform(size=len(self.transform_instances))
        logging.info("Numbers generated")
        self.print_rands()

    def print_rands(self):
        global transform_instances

        start = 0
        #pdb.set_trace()
        print("len: {}.  r1 {} r2 {} r3 {} r4 {} r5 {} r6 {}".format(
            len(transform_instances), transform_instances[start], transform_instances[start+1], transform_instances[start+2],
            transform_instances[start+3], transform_instances[start+4], transform_instances[start+5]
        ))
