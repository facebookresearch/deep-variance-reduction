# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#from __future__ import print_function
import argparse
import pickle
import os
import time
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
from diagnostics import *
from run_vr import *
import torch.backends.cudnn as cudnn

def run(args_override={}):
    run_dir = "runs"
    disable_cuda = False
    checkpoint_dir = "/checkpoint/{}/checkpoints".format(os.environ["USER"])
    default_momentum = 0.9
    default_lr = 0.1 
    default_decay = 0.0001 
    default_epochs = 300
    default_batch_size = 128 
    default_tail_average = 0.0
    default_tail_average_all = False

    default_half_precision = False
    default_method = "sgd" #"svrg" #"sgd"

    default_log_diagnostics = False
    default_log_diagnostics_every_epoch = False
    default_log_fast_diagnostics = False

    default_logfname = "log"
    default_log_interval = 20
    default_transform_locking = True

    default_per_block = False
    default_dropout = False
    default_batchnorm = True
    default_vr_from_epoch = 1 # 1 is first epoch.

    default_calculate_train_loss_each_epoch = False
    default_save_model = False # Saving every 10 epochs
    default_resume = False
    default_resume_from = ""

    # It will always resume from a checkpoint
    default_full_checkpointing = False

    default_second_lambda = 0.5 
    default_inner_steps = 10
    default_clamping = 1000.0
    default_vr_bn_at_recalibration = True 
    default_variance_reg = 0.01
    default_lr_reduction = "150-225" 
    default_L = 1.0
    default_architecture = "default"
    default_problem = "cifar10"

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch optimization testbed')
    parser.add_argument('--problem', type=str, default=default_problem,
        help='Problem instance (default: ' + default_problem + ')')
    parser.add_argument('--method', type=str, default=default_method,
        help='Optimization method (default: ' + default_method + ')')
    parser.add_argument('--batch-size', type=int,
        default=default_batch_size, metavar='M',
        help='minibatch size (default: ' + str(default_batch_size) + ')')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=default_epochs, metavar='N',
        help='number of epochs to train (default: ' + str(default_epochs) + ')')
    parser.add_argument('--lr', type=float, default=default_lr, metavar='LR',
        help='learning rate (default: ' + str(default_lr) + ')')
    parser.add_argument('--momentum', type=float, default=default_momentum,
        metavar='M',
        help='SGD momentum (default: ' + str(default_momentum) + ')')
    parser.add_argument('--decay', type=float, default=default_decay,
        metavar='M',
        help='SGD weight decay (default: ' + str(default_decay) + ')')
    parser.add_argument('--L', type=float, default=default_L,
        metavar='L',
        help='SGD L estimate (default: ' + str(default_L) + ')')

    parser.add_argument('--tail_average', type=float, default=default_tail_average,
        help='Use tail averaging of iterates every epoch, with the given tail fraction (default: ' + str(default_tail_average) + ')')
    parser.add_argument('--tail_average_all', type=str2bool, default=default_tail_average_all,
        help='Apply tail aveaging either to the whole run or just after the first lr reduction (default: ' + str(default_tail_average_all) + ')')

    parser.add_argument('--clamping', type=float, default=default_clamping,
        metavar='C', help='APS clamping (default: ' + str(default_clamping) + ')')
    parser.add_argument('--inner_steps', type=int, default=default_inner_steps, metavar='N',
        help='Inner steps for implicit methods (default: ' + str(default_inner_steps) + ')')
    parser.add_argument('--vr_from_epoch', type=int, default=default_vr_from_epoch,
        help='Start VR (if in use) at this epoch (default: ' + str(default_vr_from_epoch) + ')')

    parser.add_argument('--no-cuda', action='store_true', default=disable_cuda,
        help='disables CUDA training')
    parser.add_argument('--half_precision', type=str2bool, default=default_half_precision,
        help='Use half precision (default: ' + str(default_half_precision) + ')')

    parser.add_argument('--second_lambda', type=float, default=default_second_lambda,
        metavar='D',
        help='A second linear interpolation factor used by some algorithms (default: '
        + str(default_second_lambda) + ')')
    parser.add_argument('--variance_reg', type=float, default=default_variance_reg,
        metavar='D',
        help='Added to the variance in reparam to prevent divide by 0 problems (default: '
        + str(default_variance_reg) + ')')

    parser.add_argument('--architecture', type=str, default=default_architecture,
        help='architecture (default: ' + default_architecture + ')')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=str2bool, default=default_dropout,
        help='Use dropout (default: ' + str(default_dropout) + ')')
    parser.add_argument('--batchnorm', type=str2bool, default=default_batchnorm,
        help='Use batchnorm (default: ' + str(default_batchnorm) + ')')

    parser.add_argument('--transform_locking', type=str2bool, default=default_transform_locking,
        help='Transform locking: ' + str(default_transform_locking) + ')')

    parser.add_argument('--log_diagnostics', type=str2bool, default=default_log_diagnostics,
        help='produce and log expensive diagnostics (default: ' + str(default_log_diagnostics) + ')')

    parser.add_argument('--log_diagnostics_every_epoch', type=str2bool, default=default_log_diagnostics_every_epoch,
        help='do full diagnostics every epoch instead of every 10')

    parser.add_argument('--log_diagnostics_deciles', type=str2bool, default=False,
        help='full diagnostics at every 10% of the epoch')

    parser.add_argument('--log_fast_diagnostics', type=str2bool, default=default_log_fast_diagnostics,
        help='produce and log cheap diagnostics (default: ' + str(default_log_fast_diagnostics) + ')')

    parser.add_argument('--logfname', type=str, default=default_logfname,
        help='Prefix for diagonstic log files (default: ' + str(default_logfname) + ')')

    parser.add_argument('--save_model', type=str2bool, default=default_save_model,
        help='Save model every 10 epochs (default: ' + str(default_save_model) + ')')
    parser.add_argument('--resume', type=str2bool, default=default_resume,
        help='Resume from resume_from (default: ' + str(default_resume) + ')')
    parser.add_argument('--resume_from', type=str, default=default_resume_from,
        help=' Path to saved model (default: ' + str(default_resume_from) + ')')
    parser.add_argument('--full_checkpointing', type=str2bool, default=default_full_checkpointing,
        help='Writeout and resume from checkpoints (default: ' + str(default_full_checkpointing) + ')')
    parser.add_argument('--calculate_train_loss_each_epoch', type=str, default=default_calculate_train_loss_each_epoch,
        help=' Do a 2nd pass after each epoch to calculate the training error rate/loss (default: ' + str(default_calculate_train_loss_each_epoch) + ')')


    parser.add_argument('--vr_bn_at_recalibration', type=str2bool, default=default_vr_bn_at_recalibration,
        help='Use batch norm on the recalibration pass (default: ' + str(default_vr_bn_at_recalibration) + ')')

    parser.add_argument('--lr_reduction', type=str, default=default_lr_reduction,
        help='Use lr reduction specified (default: ' + str(default_lr_reduction) + ')')
    parser.add_argument('--log_interval', type=int, default=default_log_interval, metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--per_block', type=str2bool, default=default_per_block,
        help='Use per block learning rates (default: ' + str(default_per_block) + ')')
    args = parser.parse_args([]) # Don't actually use command line arguments, put from call to function only

    # Apply overrides?
    args.__dict__.update(args_override)

    if isinstance(args, dict):
        args = Struct(**args)

    #"scsg"
    args.opt_vr = opt_vr = (args.method in ["saga", "svrg", "pointsaga", "recompute_svrg", "online_svrg"])

    run_name = (args.problem + "-" + args.architecture + "-" +
                args.method + "-lr" + str(args.lr) +
                "-m" + str(args.momentum) + "-" + "d" + str(args.decay) +
                "-epochs" + str(args.epochs) + "bs" +
                str(args.batch_size) +
                "reduct_" + args.lr_reduction)

    if not args.batchnorm:
        run_name += "_nobn"

    if args.dropout:
        run_name += "_dropout"

    if args.opt_vr and args.vr_from_epoch != 1:
        run_name += "_vr_from_" + str(args.vr_from_epoch)

    if not args.vr_bn_at_recalibration:
        run_name += "_bn_recal_" + str(args.vr_bn_at_recalibration)

    if args.resume:
        run_name += "_resume"

    if args.seed != 1:
        run_name += "seed_" + str(args.seed)

    if args.half_precision:
        run_name += "_half"

    if args.tail_average > 0:
        run_name += "_tavg_" + str(args.tail_average)
        if args.tail_average_all:
            run_name += "_tall"

    run_name = run_name.strip().replace('.', '_')

    # SETUP LOGGING
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    ch.setFormatter(formatter)
    #if 'ch' in locals():
    root.addHandler(ch)

    ############
    logging.info("Run " + run_name)
    logging.info("#########")
    logging.info(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.info("Using CUDA: {} CUDA AVAIL: {} #DEVICES: {}".format(
        args.cuda, torch.cuda.is_available(), torch.cuda.device_count()))

    cudnn.benchmark = True

    logging.info("Loading data")
    train_loader, test_loader, model, train_dataset = problems.load(args)

    if hasattr(model, "sampler") and hasattr(model.sampler, "reorder"):
        logging.info("NOTE: Consistant batch sampling in use")

    if args.cuda:
        logging.info("model.cuda")
        model.cuda()
        logging.info("")

    if args.half_precision:
        logging.info("Using half precision")
        model = model.half()

    if args.resume:
        # Load
        model.load_state_dict(torch.load(args.resume_from, map_location=lambda storage, loc: storage))
        model.cuda()
        logging.info("Resuming from file: {}".format(args.resume_from))

    checkpoint_resume = False
    if args.full_checkpointing:
        # Look for and load checkpoint model
        checkpoint_model_path = checkpoint_dir + "/" + run_name + "_checkpoint.model"
        checkpoint_runinfo_path = checkpoint_dir + "/" + run_name + "_info.pkl"
        if os.path.exists(checkpoint_model_path):
            checkpoint_resume = True
            logging.info("Checkpoint found: {}".format(checkpoint_model_path))
            model.load_state_dict(torch.load(checkpoint_model_path, map_location=lambda storage, loc: storage))
            model.cuda()
            with open(checkpoint_runinfo_path, 'rb') as fcheckpoint:
                runinfo = pickle.load(fcheckpoint)

            if runinfo["epoch"] >= args.epochs:
                logging.error("runinfo['epoch']: {} >= args.epochs, checkpoint is past/at end of run".format(runinfo["epoch"]))
                return
        else:
            logging.info("No checkpoint exists, starting a fresh run")


    ############################
    # logging.info some information about the model
    logging.info("Model statistics:")
    nparams = 0
    group_idx = 0
    for param in model.parameters():
        #import pdb; pdb.set_trace()
        group_size = 1
        for g in param.size():
            group_size *= g
        nparams += group_size
        group_idx += 1
    train_nbatches = len(train_loader)
    logging.info("total parameters: {:,}".format(nparams))
    logging.info("minibatch size: {}".format(args.batch_size))
    logging.info("Rough training dataset size: {:,} number of minibatches: {}".format(
    len(train_loader)*args.batch_size, train_nbatches))
    logging.info("Rough test dataset size: {:,} number of test minibatches: {}".format(
    len(test_loader)*args.batch_size, len(test_loader)))

    # Averaging fraction calculation
    ntail_batches = int(train_nbatches*args.tail_average)
    if ntail_batches == 0:
        ntail_batches = 1
    ntail_from = train_nbatches - ntail_batches
    logging.info("Tail averaging fraction {:.2f} will average {} batches, from batch: {}, tail_average_all: {}".format(
        args.tail_average, ntail_batches, ntail_from, args.tail_average_all
    ))

    logging.info("Creating optimizer")
    optimizer = optimizers.optimizer(model, args)
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        model.train()
        interval = timer()
        start = timer()
        start_time = time.time()

        time_cuda = 0.0
        time_variable = 0.0
        time_forward = 0.0
        time_backward = 0.0
        time_step = 0.0
        time_load = 0.0

        if args.tail_average > 0.0:
            averaged_so_far = 0
            # create/zero tail_average storage
            for group in optimizer.param_groups:
                for p in group['params']:
                    param_state = optimizer.state[p]
                    if 'tail_average' not in param_state:
                        param_state['tail_average'] = p.data.clone().double().zero_()

        load_start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            time_load += time.time() - load_start_time

            cuda_time = time.time()
            if args.cuda:
                data, target = data.cuda(), target.cuda(non_blocking=True)
            if args.half_precision:
                data = data.half()
            variable_time = time.time()
            time_cuda += variable_time - cuda_time
            data, target = Variable(data), Variable(target)
            time_variable += time.time() - variable_time

            def eval_closure():
                nonlocal time_forward
                nonlocal time_backward
                closure_time = time.time()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                eval_time = time.time()
                time_forward += eval_time - closure_time
                loss.backward()
                time_backward += time.time() - eval_time
                return loss

            step_start_time = time.time()
            if hasattr(optimizer, "step_preds"):
                def partial_closure():
                    optimizer.zero_grad()
                    output = model(data)
                    logprobs = log_softmax(output)
                    return logprobs
                loss = optimizer.step_preds(partial_closure, target)
            elif opt_vr:
                loss = optimizer.step(batch_idx, closure=eval_closure)
            else:
                loss = optimizer.step(closure=eval_closure)

            time_step += time.time() - step_start_time

            if args.log_diagnostics and epoch >= args.vr_from_epoch:
                if args.method == "svrg":
                    in_run_diagnostics(epoch, batch_idx, args, train_loader, optimizer, model, criterion)


            # Accumulate tail average
            if args.tail_average > 0.0:
                if batch_idx >= ntail_from:
                    averaged_so_far += 1
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            param_state = optimizer.state[p]
                            tail = param_state['tail_average']
                            # Running mean calculation
                            tail.add_(1.0/averaged_so_far, p.data.double() - tail)

            if batch_idx % args.log_interval == 0:
                mid = timer()
                percent_done = 100. * batch_idx / len(train_loader)
                if percent_done > 0:
                    time_estimate = math.ceil((mid - start)*(100/percent_done))
                    time_estimate = str(datetime.timedelta(seconds=time_estimate))
                    inst_estimate =  math.ceil((mid - interval)*(len(train_loader)/args.log_interval))
                    inst_estimate = str(datetime.timedelta(seconds=inst_estimate))
                else:
                    time_estimate = "unknown"
                    inst_estimate = "unknown"
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, time: {} inst: {}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item(), time_estimate, inst_estimate))

                if False:
                    since_start = time.time()
                    logging.info("load: {:.3f}, cuda: {:.3f}, variable: {:.3f}, forward: {:.3f}, backward: {:.3f}, step: {:.3f}, step-clo: {:.3f}, sum: {}, actual: {}".format(
                        time_load, time_cuda, time_variable, time_forward, time_backward, time_step, time_step - time_forward - time_backward,
                        time_load + time_cuda + time_variable + time_step, since_start - start_time
                    ))
                    time_cuda = 0.0
                    time_variable = 0.0
                    time_forward = 0.0
                    time_backward = 0.0
                    time_step = 0.0
                    time_load = 0.0
                    interval = timer()

            load_start_time = time.time()

        if args.tail_average > 0.0:
            if averaged_so_far != ntail_batches:
                raise Exception("Off by one: {}, {}".format(averaged_so_far, ntail_batches))
            current_lr = optimizer.param_groups[0]['lr']

            if args.tail_average_all or args.lr != current_lr:
                logging.info("Setting x to tail average ({}), current_lr: {}".format(
                    args.tail_average, current_lr))

                for group in optimizer.param_groups:
                    for p in group['params']:
                        param_state = optimizer.state[p]
                        tail = param_state['tail_average']
                        p.data.zero_().add_(tail.type_as(p.data))

        if hasattr(model, "sampler") and hasattr(model.sampler, "reorder"):
            model.sampler.reorder()
        if hasattr(train_dataset, "retransform"):
            logging.info("retransform")
            train_dataset.retransform()

    def loss_stats(epoch, loader, setname):
        model.eval()
        loss = 0.0
        correct = 0.0
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            if args.half_precision:
                data = data.half()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1] # index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().float().item()

        loss /= len(loader) # loss function already averages over batch size
        error_rate = 100.0 * correct / len(loader.dataset)
        #pdb.set_trace()
        logging.info('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            setname, loss, correct, len(loader.dataset),
            error_rate))

        return (loss, error_rate)

    # Crate directory for saving model if needed
    problem_dir = run_dir + "/" + args.problem
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)

    if not checkpoint_resume:
        runinfo = vars(args)
        runinfo["train_losses"] = []
        runinfo["train_errors"] = []
        runinfo["test_losses"] = []
        runinfo["test_errors"] = []
        runinfo["nparams"] = nparams
        runinfo["ndatapoints"] = len(train_loader)*args.batch_size
        runinfo["nminibatches"] = len(train_loader)
        runinfo["epoch"] = 0

    else:
        # When resuming
        if hasattr(optimizer, "recalibrate"):
            logging.info("Recalibrate for restart, epoch: {}".format(runinfo["epoch"]))
            seed = runinfo["seed"] + 1031*runinfo["epoch"]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            recalibrate(runinfo["epoch"], args, train_loader, test_loader, model, train_dataset, optimizer, criterion)


    for epoch in range(runinfo["epoch"]+1, args.epochs + 1):
        runinfo["epoch"] = epoch
        logging.info("Starting epoch {}".format(epoch))

        seed = runinfo["seed"] + 1031*epoch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if epoch == 1 and hasattr(optimizer, "recalibrate"):
            recalibrate(epoch, args, train_loader, test_loader, model, train_dataset, optimizer, criterion)

        if args.lr_reduction == "default":
            lr = args.lr * (0.1 ** (epoch // 75))
        elif args.lr_reduction == "none" or args.lr_reduction == "False":
            lr = args.lr
        elif args.lr_reduction == "150":
            lr = args.lr * (0.1 ** (epoch // 150))
        elif args.lr_reduction == "150-225":
            lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        elif  args.lr_reduction == "up5x-20-down150":
            if epoch < 20:
                lr = args.lr
            else:
                lr = 3.0 * args.lr * (0.1 ** (epoch // 150))
        elif args.lr_reduction == "up30-150-225":
            if epoch < 30:
                lr = args.lr
            else:
                lr = 3.0 * args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        elif args.lr_reduction == "every30":
            lr = args.lr * (0.1 ** (epoch // 30))
        else:
            logging.info("Lr scheme not recognised: {}".format(args.lr_reduction))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info("Learning rate: {}".format(lr))

        start = timer()
        if args.method == "scsg":
            train_scsg(epoch, args, train_loader, test_loader, model, train_dataset, optimizer, criterion)
        else:
            train(epoch)
        end = timer()

        logging.info("Epoch took: {}".format(end-start))
        logging.info("")

        if args.calculate_train_loss_each_epoch:
            (train_loss, train_err) = loss_stats(epoch, train_loader, "Train") #test(epoch)
        else:
            train_loss = 0
            train_err = 0

        runinfo["train_losses"].append(train_loss)
        runinfo["train_errors"].append(train_err)

        (test_loss, test_err) = loss_stats(epoch, test_loader, "Test") #test(epoch)
        runinfo["test_losses"].append(test_loss)
        runinfo["test_errors"].append(test_err)

        logging.info("")

        if args.log_fast_diagnostics and hasattr(optimizer, "store_old_table"):
            logging.info("Storing old table")
            optimizer.store_old_table()
        if hasattr(optimizer, "recalibrate"):
            recalibrate(epoch+1, args, train_loader, test_loader, model, train_dataset, optimizer, criterion)
        if False: # Only works for recompute_svrg I think
            batchnorm_diagnostics(epoch, args, train_loader, optimizer, model)

        if epoch >= args.vr_from_epoch and args.log_fast_diagnostics and hasattr(optimizer, "epoch_diagnostics"):
            optimizer.epoch_diagnostics(train_loss, train_err, test_loss, test_err)

        # Ocassionally save out the model.
        if args.save_model and epoch % 10 == 0:
            logging.info("Saving model ...")
            model_dir = problem_dir + "/model_" + run_name
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_fname = "{}/epoch_{}.model".format(model_dir, epoch)
            torch.save(model.state_dict(), model_fname)
            logging.info("Saved model {}".format(model_fname))

            out_fname = problem_dir + "/" + run_name + '_partial.pkl'
            with open(out_fname, 'wb') as output:
                pickle.dump(runinfo, output)
            print("Saved partial: {}".format(out_fname))

        if args.full_checkpointing:
            checkpoint_model_path_tmp = checkpoint_model_path + ".tmp"
            logging.info("Saving checkpoint model ...")
            torch.save(model.state_dict(), checkpoint_model_path_tmp)
            os.rename(checkpoint_model_path_tmp, checkpoint_model_path)
            logging.info("Saved {}".format(checkpoint_model_path))

            checkpoint_runinfo_path_tmp = checkpoint_runinfo_path + ".tmp"
            with open(checkpoint_runinfo_path_tmp, 'wb') as output:
                pickle.dump(runinfo, output)
            os.rename(checkpoint_runinfo_path_tmp, checkpoint_runinfo_path)
            print("Saved runinfo: {}".format(checkpoint_runinfo_path))

    if True:
        if args.method == "reparm":
            optimizer.print_diagnostics()

    out_fname = problem_dir + "/" + run_name + '_final.pkl'
    with open(out_fname, 'wb') as output:
        pickle.dump(runinfo, output)
    print("Saved {}".format(out_fname))

if __name__ == "__main__":
    run()
