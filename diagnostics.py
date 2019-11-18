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

import problems
import optimizers
import logging
import pdb
from torch.nn.functional import nll_loss, log_softmax

import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#######################################################################

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def in_run_diagnostics(epoch, batch_idx, args, train_loader, optimizer, model, criterion):
    #logging.info("in run diagnostics invoked")
    if (epoch % 10) == 0 or args.log_diagnostics_every_epoch:
        nbatches = len(train_loader)
        
        if args.log_diagnostics_deciles:
            log_intervals = math.ceil(nbatches/10.0)
            log_now = batch_idx % log_intervals == 0
        else:
            lp = math.ceil(nbatches/100.0)
            log_now = batch_idx == int(math.ceil(nbatches/50.0))
            log_now = log_now or batch_idx == int(math.ceil(nbatches/9.0))
            log_now = log_now or batch_idx == int(math.ceil(nbatches/3.0))
            log_now = log_now or batch_idx == nbatches-1

        if log_now:
            print("interval, batch_idx = {}".format(batch_idx))
            optimizer.logging_pass_start()

            if optimizer.epoch >= optimizer.vr_from_epoch:
                for inner_batch_idx, (data, target) in enumerate(train_loader):
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    def eval_closure():
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        return loss

                    optimizer.logging_pass(inner_batch_idx, closure=eval_closure)
            logging.info("Logging pass finished")

            optimizer.logging_pass_end(batch_idx)

def online_svrg_diagnostics(epoch, batch_idx, args, train_loader, optimizer, model, criterion):
    if (epoch == 1 or (epoch % 10) == 0) and optimizer.epoch >= optimizer.vr_from_epoch and batch_idx == 0:
        nbatches = len(train_loader)

        mega_batch_size = optimizer.megabatch_size
        recalibration_interval = optimizer.recalibration_interval

        #print("interval, interval = {}".format(interval))
        optimizer.logging_pass_start()

        # Compute the snapshot
        snapshot_i = 0
        for inner_batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            def eval_closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss

            optimizer.snapshot_pass(inner_batch_idx, closure=eval_closure)
            snapshot_i += 1
            if snapshot_i == mega_batch_size:
                break
        logging.info("Snapshot computed")

        for interval in range(recalibration_interval):
            logging.info("Interval: {}, recal_i: {}".format(interval, optimizer.recalibration_i))

            optimizer.full_grad_init()

            # Do a full gradient calculation:
            for inner_batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                def eval_closure():
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    return loss

                optimizer.full_grad_calc(inner_batch_idx, closure=eval_closure)
            logging.info("Full grad calculation finished")

            for inner_batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                def eval_closure():
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    return loss

                optimizer.logging_pass(interval, inner_batch_idx, closure=eval_closure)
            logging.info("Logging pass finished")

            # Take a single step at the end to progress in the interval
            # Using whatever minibatch was last in the stats logging pass
            optimizer.step(inner_batch_idx, closure=eval_closure)


def scsg_diagnostics(epoch, args, train_loader, optimizer, model, criterion):
    if (epoch == 1 or (epoch % 10) == 0) and optimizer.epoch >= optimizer.vr_from_epoch:
        nbatches = len(train_loader)

        mega_batch_size = optimizer.megabatch_size
        recalibration_interval = optimizer.recalibration_interval

        #print("interval, interval = {}".format(interval))
        optimizer.logging_pass_start()

        # Compute the snapshot
        data_buffer = []
        inner_iters = optimizer.recalibration_interval
        megabatch_size = optimizer.megabatch_size
        optimizer.recalibration_i = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_id = batch_idx

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            data_buffer.append((data, target))

            # Store megabatch gradients
            def outer_closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss

            loss = optimizer.step_outer_part(closure=outer_closure)

            if len(data_buffer) == megabatch_size:
                logging.info("Snapshot complete")

                for interval in range(recalibration_interval):
                    logging.info("Interval: {}, recal_i: {}".format(interval, optimizer.recalibration_i))

                    optimizer.full_grad_init()

                    # Do a full gradient calculation:
                    for inner_batch_idx, (data, target) in enumerate(train_loader):
                        if args.cuda:
                            data, target = data.cuda(), target.cuda()
                        data, target = Variable(data), Variable(target)

                        def eval_closure():
                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            return loss

                        optimizer.full_grad_calc(closure=eval_closure)
                    logging.info("Full grad calculation finished")

                    for inner_i in range(inner_iters):
                        data, target = data_buffer[inner_i]

                        def eval_closure():
                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            return loss

                        optimizer.logging_pass(interval, closure=eval_closure)
                    logging.info("Logging pass finished")

                    # Take a single step at the end to progress in the interval
                    data, target = data_buffer[interval]

                    def eval_closure():
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        return loss

                    optimizer.step_inner_part(closure=eval_closure)

                data_buffer = []
                optimizer.recalibration_i = 0
                return

def minibatch_stats():
    # This is just copied from run.py, needs to be modified to work.
    if False:
        batch_idx, (data, target) = next(enumerate(train_loader))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        idx = 0
        ###
        optimizer.zero_grad()
        output = model(data)
        #pdb.set_trace()
        loss = criterion(output[idx, None], target[idx])
        loss.backward()

        baseline_sq = 0.0

        for group in optimizer.param_groups:
            for p in group['params']:
                gk = p.grad.data
                param_state = optimizer.state[p]
                param_state['baseline'] = gk.clone()
                baseline_sq += torch.dot(gk, gk)

        for idx in range(1, 5):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output[idx, None], target[idx])
            loss.backward()

            total_dot = 0.0
            square_norm = 0.0
            corrs = []

            for group in optimizer.param_groups:
                for p in group['params']:
                    gk = p.grad.data
                    param_state = optimizer.state[p]
                    baseline = param_state['baseline']
                    # Compute correlation
                    dp = torch.dot(baseline, gk)
                    corr = dp/(torch.norm(baseline)*torch.norm(gk))
                    corrs.append(corr)

                    total_dot += dp
                    square_norm += torch.dot(gk, gk)

            total_corr = total_dot/math.sqrt(square_norm*baseline_sq)
            logging.info("i={}, corr: {}, layers: {}".format(idx, total_corr, corrs))


        #pdb.set_trace()

def batchnorm_diagnostics(epoch, args, train_loader, optimizer, model):
    #pdb.set_trace()
    bnstuff = {'epoch': epoch, 'args': args}

    state = model.state_dict()
    for skey in state.keys():
        if skey.startswith("bn3") or skey.startswith("fc1"):
            # Convert to numpy first
            bnstuff[skey] = state[skey].cpu().numpy()
            #print("skey: {} size: {}".format(skey, state[skey].size()))

            # Search optimizer state for param_state for this variable
            for group in optimizer.param_groups:
                for p in group['params']:
                    gk = p.grad.data
                    param_state = optimizer.state[p]
                    if id(p.data) == id(state[skey]):
                        #print("match")
                        bnstuff[skey + ".gavg"] = param_state["gavg"].cpu().numpy()

    # Store to disk I guess
    fname = 'stats/{}_batchnorm_epoch{}.pkl'.format(args.logfname, epoch)
    with open(fname, 'wb') as output:
        pickle.dump(bnstuff, output)
    logging.info("Wrote out batch norm stats: {}".format(fname))
