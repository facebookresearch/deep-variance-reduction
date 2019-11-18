# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
import torchvision
import torchvision.transforms as transforms
import logging
from torch.autograd import Variable
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist

from cifar_wrapper import CIFAR10_Wrapper
from vr_sampler import VRSampler
import UpdatedDataLoader
import UpdatedDataLoaderMult
import resnet
import pdb
import densenet
import resnext
from imagenet_wrapper import ImagenetWrapper

def load(args):
    print("Problem:", args.problem)
    if args.problem == "cifar10":
        return cifar10(args)
    elif args.problem == "imagenet":
        return imagenet(args)
    else:
        raise Exception("Unrecognised problem:", args.problem)


def cifar10(args):
    data_dir = os.path.expanduser('~/data')
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    transform = transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # We don't do the random transforms at test time.
    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    logging.info("Loading training dataset")
    if (args.method.endswith("svrg") or args.method == "scsg") and args.transform_locking and args.opt_vr:
        train_dataset = CIFAR10_Wrapper(
            root=data_dir, train=True,
            download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform)

    if args.method.endswith("svrg") and args.opt_vr:
        if args.method == "saga":
            raise Exception("vr sampler currently doesn't support saga")
        logging.info("VR Sampler with order=perm")
        sampler = VRSampler(order="perm",
            batch_size=args.batch_size,
            dataset_size=len(train_dataset))
        train_loader = UpdatedDataLoader.DataLoader(
          train_dataset, batch_sampler=sampler, **kwargs)
    else:
        sampler = RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
          train_dataset, sampler=sampler, batch_size=args.batch_size, **kwargs)

    args.nbatches = len(sampler)

    logging.info("Loading test dataset")
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, **kwargs)

    nonlinearity = F.relu

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            logging.info("Initializing fully connected layers")
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

            if args.batchnorm:
                logging.info("Using batchnorm")
                self.bn1 = nn.BatchNorm2d(6)
                self.bn2 = nn.BatchNorm2d(16)
                self.bn3 = nn.BatchNorm1d(120)
                self.bn4 = nn.BatchNorm1d(84)
            logging.info("initialized")

        def forward(self, x):
            x = self.conv1(x)
            if args.batchnorm:
                x = self.bn1(x)
            x = nonlinearity (x)
            x = self.pool(x)
            #pdb.set_trace()

            x = self.conv2(x)
            if args.batchnorm:
                x = self.bn2(x)
            x = nonlinearity (x)
            x = self.pool(x)

            x = x.view(-1, 16 * 5 * 5)
            x = self.fc1(x)

            if args.batchnorm:
                x = self.bn3(x)
            x = nonlinearity (x)
            x = self.fc2(x)

            if args.batchnorm:
                x = self.bn4(x)
            x = nonlinearity (x)
            x = self.fc3(x)
            return x

    logging.info("Loading architecture")
    if args.architecture == "default":
        logging.info("default architecture")
        model = Net()
    elif args.architecture == "resnet110":
        model = resnet.ResNet110(batchnorm=args.batchnorm, nonlinearity=nonlinearity)
    elif args.architecture == "resnet-small":
        model = resnet.ResNetSmall(batchnorm=args.batchnorm, nonlinearity=nonlinearity)
    elif args.architecture == "densenet-40-36":
        model = densenet.densenet(depth=40, growthRate=36, batchnorm=args.batchnorm, nonlinearity=nonlinearity)
        model = torch.nn.DataParallel(model)
    else:
        raise Exception("architecture not recognised:", args.architecture)

    model.sampler = sampler
    return train_loader, test_loader, model, train_dataset

def imagenet(args):
    kwargs = {'num_workers': 32, 'pin_memory': True} if args.cuda else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    lock_transforms = (args.method.endswith("svrg")) and args.transform_locking and args.opt_vr

    logging.info("Loading training dataset")
    train_dir = "/datasets01_101/imagenet_full_size/061417/train"

    logging.info("Data ...")
    train_dataset = ImagenetWrapper(train_dir, lock_transforms=lock_transforms)
    logging.info("Imagenet Wrapper created")

    logging.info("VR Sampler with order=perm")
    sampler = VRSampler(order="perm",
        batch_size=args.batch_size,
        dataset_size=len(train_dataset))

    train_loader = UpdatedDataLoaderMult.DataLoader(
        train_dataset, batch_sampler=sampler,
        worker_init_fn=train_dataset.child_initialize, **kwargs) #worker_init_fn
    logging.info("Train Loader created, batches: {}".format(len(train_loader)))

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("/datasets01_101/imagenet_full_size/061417/val",
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    args.nbatches = len(train_loader)

    logging.info("Initializing model")
    if args.architecture == "resnet18":
        model = torchvision.models.resnet.resnet18()
    elif args.architecture == "resnet50":
        model = torchvision.models.resnet.resnet50()
    elif args.architecture == "resnext101_32x8d":
        model = resnext.resnext101_32x8d()
    else:
        raise Exception("Architecture not supported for imagenet")

    logging.info("Lifting model to DataParallel")
    model = torch.nn.DataParallel(model).cuda() # Use multiple gpus
    model.sampler = sampler

    return train_loader, test_loader, model, train_dataset
