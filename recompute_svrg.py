# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torch.optim.optimizer import Optimizer, required
import torch
import pdb
import pickle
import math
import logging

class RecomputeSVRG(Optimizer):
    r"""Implements the standard SVRG method
    """

    def __init__(self, params, nbatches, model, vr_bn_at_recalibration,
                 vr_from_epoch,
                 lr=required, momentum=required, weight_decay=required):
        self.nbatches = nbatches
        self.batches_processed = 0
        self.epoch = 0
        self.vr_bn_at_recalibration = vr_bn_at_recalibration
        self.vr_from_epoch = vr_from_epoch
        self.model = model
        self.running_tmp = {}
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(RecomputeSVRG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RecomputeSVRG, self).__setstate__(state)

    def initialize(self):
        for group in self.param_groups:
            for p in group['params']:
                momentum = group['momentum']

                param_state = self.state[p]

                if 'gavg' not in param_state:
                    param_state['gavg'] =  p.data.double().clone().zero_()
                    param_state['gi'] = p.data.clone().zero_()
                    param_state['gi_debug'] = p.data.clone().zero_()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.clone().zero_()

                if 'tilde_x' not in param_state:
                    param_state['tilde_x'] = p.data.clone()
                    param_state['xk'] = p.data.clone()

        # Batch norm's activation running_mean/var variables
        state = self.model.state_dict()
        for skey in state.keys():
            if skey.endswith(".running_mean") or skey.endswith(".running_var"):
                self.running_tmp[skey] = state[skey].clone()

    def recalibrate_start(self):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """
        self.epoch += 1
        self.recal_calls = 0
        self.initialize()
        self.store_running_mean()
        print("Recal epoch: {}".format(self.epoch))

        if self.epoch >= self.vr_from_epoch:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    gavg = param_state['gavg']
                    gavg.zero_()

                    tilde_x = param_state['tilde_x']
                    tilde_x.zero_().add_(p.data.clone())
                    #pdb.set_trace()
        else:
            logging.info("Skipping recalibration as epoch {} not >= {}".format(
            self.epoch, self.vr_from_epoch))


    def recalibrate(self, batch_id, closure):
        """ Compute part of the full batch gradient, from one minibatch
        """
        loss = closure()

        if self.epoch >= self.vr_from_epoch:
            self.recal_calls += 1
            for group in self.param_groups:
                for p in group['params']:
                    gk = p.grad.data

                    param_state = self.state[p]

                    gavg = param_state['gavg']
                    gavg.add_(1.0/self.nbatches, gk.double())

        return loss

    def recalibrate_end(self):
        self.restore_running_mean()
        if self.recal_calls != self.nbatches:
            raise Exception("recalibrate_end called, with {} nbatches: {}".format(
                            self.recal_calls, self.nbatches))

    def epoch_diagnostics(self, train_loss, train_err, test_loss, test_err):
            """
            Called after recalibrate, saves stats out to disk.
            """


    def step(self, batch_id, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if self.epoch >= self.vr_from_epoch:
            self.store_running_mean()
            ## Store current xk, replace with x_tilde
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    xk = param_state['xk']
                    xk.zero_().add_(p.data)
                    p.data.zero_().add_(param_state['tilde_x'])

            # Standard is vr_bn_at_recalibration=True, so this doesn't fire.
            if not self.vr_bn_at_recalibration:
                self.model.eval() # turn off batch norm
            ## Compute gradient at x_tilde
            closure()

            ## Store x_tilde gradient in gi, and revert to xk
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    xk = param_state['xk']
                    gi = param_state['gi']
                    gi.zero_().add_(p.grad.data)
                    p.data.zero_().add_(xk)

            # Make sure batchnorm is handled correctly.
            self.restore_running_mean()

        ## compute gradient at xk
        loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            learning_rate = group['lr']

            for p in group['params']:
                gk = p.grad.data

                param_state = self.state[p]

                gi = param_state['gi']
                gavg = param_state['gavg']

                if momentum != 0:
                    buf = param_state['momentum_buffer']

                #########

                if self.epoch >= self.vr_from_epoch:
                    vr_gradient = gk.clone().sub_(gi).add_(gavg.type_as(gk))
                else:
                    vr_gradient = gk.clone() # Just do sgd steps

                if weight_decay != 0:
                    vr_gradient.add_(weight_decay, p.data)

                if momentum != 0:
                    dampening = 0.0
                    vr_gradient = buf.mul_(momentum).add_(1 - dampening, vr_gradient)

                # Take step.
                p.data.add_(-learning_rate, vr_gradient)


        # track number of minibatches seen
        self.batches_processed += 1

        return loss

    def store_running_mean(self):
        # Store running_mean/var temporarily
        state = self.model.state_dict()
        #pdb.set_trace()
        for skey in self.running_tmp.keys():
            self.running_tmp[skey].zero_().add_(state[skey])

    def restore_running_mean(self):
        state = self.model.state_dict()
        for skey in self.running_tmp.keys():
            state[skey].zero_().add_(self.running_tmp[skey])
