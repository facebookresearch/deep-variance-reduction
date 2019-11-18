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

class SCSG(Optimizer):
    def __init__(self, params, args, nbatches, model, vr_bn_at_recalibration,
                 vr_from_epoch,
                 lr=required, momentum=required, weight_decay=required):
        self.nbatches = nbatches
        self.batches_processed = 0
        self.epoch = 0
        self.vr_bn_at_recalibration = vr_bn_at_recalibration
        self.vr_from_epoch = vr_from_epoch
        self.model = model
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.recompute_version = False
        self.megabatch_size = 10 # number of minibatches in a megabatch
        self.recalibration_interval = 10
        self.recalibration_i = 0
        self.interval_i = 0
        self.stats_buffered = False

        if self.megabatch_size != self.recalibration_interval:
            raise Exception("megabatch_size != recalibration_interval not supported yet")

        self.test_name = args.logfname

        self.running_tmp = {}
        self.running_interp = 0.9

        super(SCSG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SCSG, self).__setstate__(state)

    def initialize(self):
        for group in self.param_groups:
            for p in group['params']:
                momentum = group['momentum']

                param_state = self.state[p]

                if 'gavg' not in param_state:
                    param_state['gavg'] =  p.data.clone().zero_()
                    param_state['gavg_debug'] =  p.data.clone().zero_()
                    param_state['full_grad'] =  p.data.clone().zero_()
                    param_state['gi'] = p.data.clone().zero_()

                    if not self.recompute_version:
                        gsize = p.data.size()
                        gtbl_size = torch.Size([self.megabatch_size] + list(gsize))
                        param_state['gktbl'] = torch.zeros(gtbl_size).cuda()

                    # m2 is the running gradient variance accumulator
                    param_state['m2'] = p.data.clone().zero_()

                    param_state['grad_running_cov'] = p.data.clone().zero_()
                    param_state['grad_running_var'] = p.data.clone().zero_()
                    param_state['grad_running_mean'] = p.data.clone().zero_()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.clone().zero_()

                if 'tilde_x' not in param_state:
                    param_state['tilde_x'] = p.data.clone()
                    param_state['xk'] = p.data.clone()


        state = self.model.state_dict()
        # Batch norm's activation running_mean/var variables
        for skey in state.keys():
            if skey.endswith(".running_mean") or skey.endswith(".running_var"):
                self.running_tmp[skey] = state[skey].clone()

        logging.info("running: {}".format(self.running_tmp.keys()))

    def recalibrate_start(self):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use. We don't do anything in online_svrg
        """
        print("Recalibrate_start called")
        self.epoch += 1
        self.initialize()

        gi_var = [0.0 for i in range(self.recalibration_interval)]
        vr_var = [0.0 for i in range(self.recalibration_interval)]

        if self.stats_buffered:
            # Write out any variance statistics
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    #fg = param_state['full_grad']
                    #fg.zero_() # reset the full gradient
                    for i in range(self.recalibration_interval):
                        gi_var[i] += param_state["gi_var_acum"][i].sum()/self.recalibration_interval
                        vr_var[i] += param_state["vr_var_acum"][i].sum()/self.recalibration_interval

            fname = 'stats/{}_scsg_{}.pkl'.format(self.test_name, self.epoch)
            with open(fname, 'wb') as output:
                pickle.dump({
                    'gi_var': gi_var,
                    'vr_var': vr_var,
                    'epoch': self.epoch,
                }, output)
            #self.gradient_variances = []
            #self.vr_step_variances = []
            #self.batch_indices = []

            self.stats_buffered = False
            print("logging pass diagnostics saved to {}".format(fname))



    def recalibrate(self, batch_id, closure):
        """ SCSG doesn't use recalibration passes.
        """
        return 0.0

    def epoch_diagnostics(self, train_loss, train_err, test_loss, test_err):
            """
            Called after recalibrate, saves stats out to disk.
            """


    def step_outer_part(self, closure, idx):
        self.store_running_mean()
        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                gk = p.grad.data
                param_state = self.state[p]
                gavg = param_state['gavg']
                gavg_debug = param_state['gavg_debug']

                if self.recalibration_i == 0:
                    param_state['tilde_x'].zero_().add_(p.data)
                    gavg.zero_()
                    gavg_debug.zero_()
                    param_state['m2'].zero_()

                gavg_debug.add_(1.0/self.megabatch_size, gk)

                if not self.recompute_version:
                    param_state['gktbl'][idx, :] = gk

                m2 = param_state['m2']

                # Online mean/variance calcuation from wikipedia
                delta = gk - gavg
                gavg.add_(1.0/(self.recalibration_i+1), delta)
                #if self.recalibration_i == 4:
                #    pdb.set_trace()
                delta2 = gk - gavg
                m2.add_(delta*delta2)

        self.restore_running_mean()
        self.recalibration_i += 1
        #self.batches_processed += 1
        return loss

    def step_inner_part(self, closure, idx):
        # Check a few things:
        if self.recalibration_i != self.megabatch_size:
            raise Exception("bad self.recalibration_i: {}".format(self.recalibration_i))

        if self.recompute_version:
            self.store_running_mean()

            ## Store current xk, replace with x_tilde
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    xk = param_state['xk']
                    xk.zero_().add_(p.data)
                    p.data.zero_().add_(param_state['tilde_x'])

            ## Compute gradient at x_tilde
            closure()

            ## Store x_tilde gradient in gi, and revert to xk
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    xk = param_state['xk']
                    param_state['gi'].zero_().add_(p.grad.data)
                    p.data.zero_().add_(xk)

            self.restore_running_mean()

            # JUST FOR DEBUGGING
            if False:
                for group in self.param_groups:
                    for p in group['params']:
                        param_state = self.state[p]
                        gi = param_state['gi']
                        gi_tbl = param_state['gktbl'][idx, :]
                        #pdb.set_trace()
                        if torch.norm(gi-gi_tbl) > 1e-6:
                            print("difference: {}".format( torch.norm(gi-gi_tbl)))
                            pdb.set_trace()
        else:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['gi'] = param_state['gktbl'][idx, :]

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
                #gavg_debug = param_state['gavg_debug']

                if momentum != 0:
                    buf = param_state['momentum_buffer']

                #########

                if self.epoch < self.vr_from_epoch:
                    vr_gradient = gk.clone() # Just do sgd steps
                else:
                    vr_gradient = gk.clone().sub_(gi).add_(gavg)

                    # Track the running mean and variance of the gradients.
                    grad_running_mean = param_state['grad_running_mean']
                    grad_running_var = param_state['grad_running_var']
                    grad_running_cov = param_state['grad_running_cov']
                    cov_update = (gk - grad_running_mean)*(gi - gavg)
                    grad_running_cov.mul_(self.running_interp).add_(1-self.running_interp, cov_update)
                    # Using delta from before and after the mean update is apparently the
                    # best way to update variances.
                    delta1 = gk - grad_running_mean
                    grad_running_mean.mul_(self.running_interp).add_(1-self.running_interp, gk)
                    delta2 = gk - grad_running_mean
                    var_update = delta1*delta2
                    grad_running_var.mul_(self.running_interp).add_(1-self.running_interp, var_update)


                    #if torch.norm(gavg-gavg_debug) > 1e-7:
                    #    raise Exception("gavg norm diff: {}".format(torch.norm(gavg-gavg_debug)))

                if weight_decay != 0:
                    vr_gradient.add_(weight_decay, p.data)

                if momentum != 0:
                    dampening = 0.0
                    vr_gradient = buf.mul_(momentum).add_(1 - dampening, vr_gradient)

                # Take step.
                p.data.add_(-learning_rate, vr_gradient)


        # track number of minibatches seen
        #logging.info("interval i: {}".format(self.interval_i))
        self.batches_processed += 1

        if self.batches_processed % 20 == 0 and self.batches_processed > 0:
            running_cov_acum = 0.0
            m2_acum = 0.0
            var_acum = 0.0
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    grad_running_cov = param_state['grad_running_cov']
                    grad_running_var = param_state['grad_running_var']
                    m2 = param_state['m2']
                    running_cov_acum += grad_running_cov.sum()
                    var_acum += grad_running_var.sum()
                    # m2 is not stored normalized by self.nbatches
                    m2_norm = m2.div(self.megabatch_size)
                    m2_acum += m2_norm.sum()

            if m2_acum > 0:
                cov_var_ratio = running_cov_acum/m2_acum

                vr_variance = var_acum + m2_acum - 2*running_cov_acum
                vr_ratio = vr_variance/var_acum
                corr_coef = running_cov_acum/math.sqrt(var_acum*m2_acum)
                logging.info("VR RATIO: {:.3f}. Raw cov/var: {:.3f}, correlation coef: {:.3f}. Var: {:.3f} m2: {:.3f} cov: {:.3f}".format(
                    vr_ratio, cov_var_ratio, corr_coef, var_acum, m2_acum, running_cov_acum))
        return loss

    ###############################################
    ###############################################
    def full_grad_init(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                fg = param_state['full_grad']
                fg.zero_()

    def full_grad_calc(self, closure):
        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                gk = p.grad.data
                param_state = self.state[p]
                fg = param_state['full_grad']
                fg.add_(1.0/self.nbatches, gk)

    def logging_pass_start(self):
        self.recalibration_i = 0
        self.interval_i = 0
        self.stats_buffered = True

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                pgz = p.grad.data.clone().zero_()
                #param_state["gi_mean_acum"] = []
                param_state["gi_var_acum"] = []
                #param_state["vr_mean_acum"] = []
                param_state["vr_var_acum"] = []
                for i in range(self.recalibration_interval):
                    #param_state["gi_mean_acum"].append(pgz.clone())
                    param_state["gi_var_acum"].append(pgz.clone())
                    #param_state["vr_mean_acum"].append(pgz.clone())
                    param_state["vr_var_acum"].append(pgz.clone())

    def logging_pass(self, interval, closure):
        self.store_running_mean()

        ## Store current xk, replace with x_tilde
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                xk = param_state['xk']
                xk.zero_().add_(p.data)
                p.data.zero_().add_(param_state['tilde_x'])

        ## Compute gradient at x_tilde
        closure()

        ## Store x_tilde gradient in gi, and revert to xk
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                xk = param_state['xk']
                param_state['gi'].zero_().add_(p.grad.data)
                p.data.zero_().add_(xk)

        # Restore running_mean/var
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
                full_grad = param_state['full_grad']
                gi = param_state['gi']
                gavg = param_state['gavg']

                vr_gradient = gk.clone().sub_(gi).add_(gavg)

                # Online mean/variance calcuation
                delta = gk - full_grad
                param_state["gi_var_acum"][interval].add_(delta*delta)
                # var version
                delta = vr_gradient - full_grad
                param_state["vr_var_acum"][interval].add_(delta*delta)

        return loss

    def store_running_mean(self):
        # Store running_mean/var temporarily
        state = self.model.state_dict()
        for skey in self.running_tmp.keys():
            self.running_tmp[skey].zero_().add_(state[skey])

    def restore_running_mean(self):
        state = self.model.state_dict()
        for skey in self.running_tmp.keys():
            state[skey].zero_().add_(self.running_tmp[skey])
