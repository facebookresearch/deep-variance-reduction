# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import argparse
import pickle
import os
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import math
import datetime

import problems
import optimizers
import logging
import pdb
from torch.nn.functional import nll_loss, log_softmax
import numpy as np

def recalibrate(epoch, args, train_loader, test_loader, model, train_dataset, optimizer, criterion):
    if args.vr_bn_at_recalibration:
        model.train()
    else:
        model.eval()
    logging.info("Recalibration pass starting")
    if hasattr(optimizer, "recalibrate_start"):
        optimizer.recalibrate_start()
    start = timer()

    #logging.info("Recalibration loop ...")
    if optimizer.epoch >= optimizer.vr_from_epoch and args.method != "online_svrg" and args.method != "scsg":
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_id = batch_idx
            #pdb.set_trace()
            if args.cuda:
                data, target = data.cuda(), target.cuda(non_blocking=True)
            data, target = Variable(data), Variable(target)

            #print("recal:")
            #print(data[:2].data.cpu().numpy())

            def eval_closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss

            optimizer.recalibrate(batch_id, closure=eval_closure)

            if batch_idx % args.log_interval == 0:
                mid = timer()
                percent_done = 100. * batch_idx / len(train_loader)
                if percent_done > 0:
                    time_estimate = math.ceil((mid - start)*(100/percent_done))
                    time_estimate = str(datetime.timedelta(seconds=time_estimate))
                else:
                    time_estimate = "unknown"

                logging.info('Recal Epoch: {} [{}/{} ({:.0f}%)] estimate: {}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    percent_done, time_estimate))

    if hasattr(optimizer, "recalibrate_end"):
        optimizer.recalibrate_end()
    logging.info("Recalibration finished")

def train_scsg(epoch, args, train_loader, test_loader, model, train_dataset, optimizer, criterion):
    logging.info("Train (SCSG version)")
    model.train()

    data_buffer = []
    inner_iters = optimizer.recalibration_interval
    megabatch_size = optimizer.megabatch_size
    optimizer.recalibration_i = 0
    logged = False

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # Store megabatch gradients
        def outer_closure():
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss

        loss = optimizer.step_outer_part(closure=outer_closure, idx=len(data_buffer))
        data_buffer.append((data, target))

        # When data-buffer is full, do the actual inner steps.
        if len(data_buffer) == megabatch_size:

            for inner_i in range(inner_iters):
                data, target = data_buffer[inner_i]

                def eval_closure():
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    return loss

                optimizer.step_inner_part(closure=eval_closure, idx=inner_i)

            data_buffer = []
            optimizer.recalibration_i = 0

            if not logged and args.log_diagnostics and epoch >= args.vr_from_epoch:
                scsg_diagnostics(epoch, args, train_loader, optimizer, model, criterion)
                logged = True


        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


    if hasattr(model, "sampler") and hasattr(model.sampler, "reorder"):
        model.sampler.reorder()
    if hasattr(train_dataset, "retransform"):
        logging.info("retransform")
        train_dataset.retransform()
