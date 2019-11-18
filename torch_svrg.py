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
import scipy
import scipy.stats
import scipy.stats.mstats

class SVRG(Optimizer):
    r"""Implements the standard SVRG method
    """

    def __init__(self, params, args, nbatches, lr=required,
                 momentum=required, weight_decay=required):
        self.nbatches = nbatches
        self.batches_processed = 0
        self.epoch = 0
        self.vr_from_epoch = args.vr_from_epoch

        self.test_name = args.logfname #"densenet" #"resnet-" #"sgd-" #"resnet-"
        if args.transform_locking:
            self.test_name += "_LOCK_"
        else:
            self.test_name += "_ULOCK_"

        self.recalibration_i = 0
        self.running_interp = 0.9
        self.denom_epsilon = 1e-7 # avoid divide by zero

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.gradient_variances = []
        self.vr_step_variances = []
        self.batch_indices = []
        self.iterate_distances = []

        self.inrun_iterate_distances = []
        self.inrun_grad_distances = []

        super(SVRG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)

    def initialize(self):
        m = self.nbatches

        for group in self.param_groups:

            for p in group['params']:
                momentum = group['momentum']

                gsize = p.data.size()
                gtbl_size = torch.Size([m] + list(gsize))

                param_state = self.state[p]

                if 'gktbl' not in param_state:
                    param_state['gktbl'] = torch.zeros(gtbl_size)
                    param_state['logging_gktbl'] = torch.zeros(gtbl_size)

                if 'tilde_x' not in param_state:
                    param_state['tilde_x'] = p.data.clone()
                    param_state['running_x'] = p.data.clone()

                if 'gavg' not in param_state:
                    param_state['gavg'] = p.data.clone().double().zero_()
                    param_state['logging_gavg'] = p.data.clone().double().zero_()
                    param_state['m2'] = p.data.clone().double().zero_()
                    param_state['running_cov'] = p.data.clone().double().zero_()
                    param_state['running_mean'] = p.data.clone().double().zero_()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.clone().zero_()

    def store_old_table(self):
        """
        Stores the old gradient table for recalibration purposes.
        """

        for group in self.param_groups:
            for p in group['params']:
                gk = p.grad.data

                param_state = self.state[p]

                gktbl = param_state['gktbl']
                gavg = param_state['gavg']

                param_state['gktbl_old'] = gktbl.clone()
                param_state['gavg_old'] = gavg.clone()


    def recalibrate_start(self):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """
        self.epoch += 1
        self.initialize()
        self.recalibration_i = 0

        # Write out any logging stats if needed
        if len(self.gradient_variances) > 0:
            fname = 'stats/{}variance_epoch{}.pkl'.format(self.test_name, self.epoch)
            with open(fname, 'wb') as output:
                pickle.dump({
                    'gradient_variances': self.gradient_variances,
                    'vr_step_variances': self.vr_step_variances,
                    'batch_indices': self.batch_indices,
                    'iterate_distances': self.iterate_distances,
                    'epoch': self.epoch,
                }, output)
            self.gradient_variances = []
            self.vr_step_variances = []
            self.batch_indices = []
            print("logging pass diagnostics saved to {}".format(fname))

        if self.epoch >= self.vr_from_epoch:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['gavg'].zero_()
                    param_state['m2'].zero_()

                    # xk is changed to the running_x
                    p.data.zero_().add_(param_state['running_x'])
                    param_state['tilde_x'] = p.data.clone()

        else:
            logging.info("Skipping recalibration as epoch {} not >= {}".format(
            self.epoch, self.vr_from_epoch))

    def recalibrate(self, batch_id, closure):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """
        loss = closure()
        #print("recal loss:", loss)

        m = self.nbatches
        self.recalibration_i += 1

        if self.epoch >= self.vr_from_epoch:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        print("grad none")
                        pdb.set_trace()
                        continue
                    gk = p.grad.data.double()

                    param_state = self.state[p]

                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg']
                    m2 = param_state['m2']
                    #pdb.set_trace()

                    # Online mean/variance calcuation from wikipedia
                    delta = gk - gavg
                    gavg.add_(1.0/self.recalibration_i, delta)
                    delta2 = gk - gavg
                    m2.add_((delta*delta2).type_as(m2))

                    param_state['running_mean'].zero_().add_(gavg)
                    param_state['running_cov'].zero_().add_(1.0/self.nbatches, m2.double())

                    #########
                    gktbl[batch_id, :] = p.grad.data.cpu().clone()

        return loss

    def epoch_diagnostics(self, train_loss, train_err, test_loss, test_err):
        """
        Called after recalibrate, saves stats out to disk.
        """
        m = self.nbatches
        logging.info("Epoch diagnostics computation")

        layernum = 0
        layer_gradient_norm_sqs = []
        gavg_norm_acum = 0.0
        gavg_acum = []
        for group in self.param_groups:
            for p in group['params']:

                layer_gradient_norm_sqs.append([])
                gavg = self.state[p]['gavg'].cpu()
                gavg_acum.append(gavg.numpy())
                gavg_norm_acum += gavg.norm()**2 #torch.dot(gavg, gavg)
                layernum += 1

        gradient_norm_sqs = []
        vr_step_variance = []
        cos_acums = []
        variances = []

        for batch_id in range(m):
            norm_acum = 0.0
            ginorm_acum = 0.0
            vr_acum = 0.0
            layernum = 0
            cos_acum = 0.0
            var_acum = 0.0
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]

                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data).cpu()

                    gi = gktbl[batch_id, :]
                    var_norm_sq = (gi-gavg).norm()**2 #torch.dot(gi-gavg, gi-gavg)
                    norm_acum += var_norm_sq
                    ginorm_acum += gi.norm()**2 #torch.dot(gi, gi)
                    layer_gradient_norm_sqs[layernum].append(var_norm_sq)

                    gktbl_old = param_state['gktbl_old']
                    gavg_old = param_state['gavg_old'].type_as(p.data).cpu()
                    gi_old = gktbl_old[batch_id, :]
                    #pdb.set_trace()
                    vr_step = gi - gi_old + gavg_old
                    vr_acum += (vr_step - gavg).norm()**2 #torch.dot(vr_step - gavg, vr_step - gavg)
                    cos_acum += torch.sum(gavg*gi)

                    var_acum += (gi - gavg).norm()**2

                    layernum += 1
            gradient_norm_sqs.append(norm_acum)
            vr_step_variance.append(vr_acum)
            cosim = cos_acum/math.sqrt(ginorm_acum*gavg_norm_acum)
            #pdb.set_trace()
            cos_acums.append(cosim)
            variances.append(var_acum)

        variance = sum(variances)/len(variances)

        print("mean cosine: {}".format(sum(cos_acums)/len(cos_acums)))

        #pdb.set_trace()

        with open('stats/{}fastdiagnostics_epoch{}.pkl'.format(self.test_name, self.epoch), 'wb') as output:
            pickle.dump({
                'train_loss': train_loss,
                'train_err': train_err,
                'test_loss': test_loss,
                'test_err': test_err,
                'epoch': self.epoch,
                #'layer_gradient_norm_sqs': layer_gradient_norm_sqs,
                #'gradient_norm_sqs': gradient_norm_sqs,
                #'vr_step_variance': vr_step_variance,
                #'cosine_distances': cos_acums,
                #'variances': variances,
                'variance': variance,
                #'gavg_norm': gavg_norm_acum,
                #'gavg': gavg_acum,
                #'iterate_distances': self.inrun_iterate_distances,
                #'grad_distances': self.inrun_grad_distances,
            }, output)
        print("Epoch diagnostics saved")
        #pdb.set_trace()

        self.inrun_iterate_distances = []
        self.inrun_grad_distances = []

    def logging_pass_start(self):
        self.logging_evals = 0

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                logging_gavg = param_state['logging_gavg']
                logging_gavg.zero_()


    def logging_pass(self, batch_id, closure):
        loss = closure()
        m = self.nbatches

        for group in self.param_groups:
            for p in group['params']:
                gk = p.grad.data

                param_state = self.state[p]

                logging_gktbl = param_state['logging_gktbl']
                logging_gavg = param_state['logging_gavg']

                logging_gavg.add_(1.0/m, gk.double())
                logging_gktbl[batch_id, :] = gk.cpu().clone()

        self.logging_evals += 1
        return loss

    def logging_pass_end(self, batch_idx):
        m = self.nbatches
        logging.info("logging diagnostics computation")

        gradient_sqs = []
        vr_step_sqs = []
        forth_sqs = []
        dist_sq_acum = 0.0


        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                tilde_x = param_state['tilde_x']
                iterate_diff = p.data - tilde_x
                dist_sq_acum += iterate_diff.norm()**2 #torch.dot(iterate_diff,iterate_diff)

        dist = math.sqrt(dist_sq_acum)

        for batch_id in range(m):
            grad_norm_acum = 0.0
            vr_norm_acum = 0.0
            forth_acum = 0.0
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]

                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data).cpu()
                    gi = gktbl[batch_id, :].type_as(p.data).cpu()

                    # Logging versions are at current location xk,
                    # compared to gavg/tktbl which are at xtilde
                    logging_gktbl = param_state['logging_gktbl']
                    logging_gavg = param_state['logging_gavg'].type_as(p.data).cpu()
                    logging_gi = logging_gktbl[batch_id, :].type_as(p.data).cpu()

                    vr_step = (logging_gi - gi + gavg) - logging_gavg
                    gi_step = logging_gi - logging_gavg
                    grad_norm_acum += gi_step.pow(2.0).sum().item()
                    vr_norm_acum += vr_step.pow(2.0).sum().item()
                    forth_acum += gi_step.pow(2.0).sum().item()
            gradient_sqs.append(grad_norm_acum)
            vr_step_sqs.append(vr_norm_acum)
            forth_sqs.append(forth_acum**2)

        # Compute variance numbers
        gradient_variance = sum(gradient_sqs)/m
        fourth_moment = sum(forth_sqs)/m - gradient_variance**2
        vr_step_variance = sum(vr_step_sqs)/m
        logging.info("gradient variance: {} vr: {}, ratio vr/g: {}".format(
            gradient_variance, vr_step_variance, vr_step_variance/gradient_variance))
        logging.info(f"forth: {fourth_moment} relative std: {math.sqrt(fourth_moment)/gradient_variance} rel SE: {math.sqrt(fourth_moment/m)/gradient_variance}")
        logging.info("self.logging_evals: {}".format(self.logging_evals))
        #pdb.set_trace()

        self.gradient_variances.append(gradient_variance)
        self.vr_step_variances.append(vr_step_variance)
        self.batch_indices.append(batch_idx)
        self.iterate_distances.append(dist)

    def step(self, batch_id, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = closure()
        dist_sq_acum = 0.0
        grad_dist_sq_acum = 0.0

        #print("step loss: ", loss)

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            learning_rate = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                gk = p.grad.data

                param_state = self.state[p]

                gktbl = param_state['gktbl']
                gavg = param_state['gavg'].type_as(p.data)
                tilde_x = param_state['tilde_x']

                if momentum != 0:
                    buf = param_state['momentum_buffer']

                #########

                if self.epoch < self.vr_from_epoch:
                    vr_gradient = gk.clone() # Just do sgd steps
                else:
                    gi = gktbl[batch_id, :].cuda()

                    vr_gradient = gk.clone().sub_(gi - gavg)

                    # Some diagnostics
                    iterate_diff = p.data - tilde_x
                    #pdb.set_trace()
                    dist_sq_acum += iterate_diff.norm()**2 #torch.dot(iterate_diff,iterate_diff)
                    grad_diff = gi - gk
                    grad_dist_sq_acum += grad_diff.norm()**2 #torch.dot(grad_diff,grad_diff)

                if weight_decay != 0:
                    vr_gradient.add_(weight_decay, p.data)

                if momentum != 0:
                    dampening = 0.0
                    vr_gradient = buf.mul_(momentum).add_(1 - dampening, vr_gradient)

                # Take step.
                p.data.add_(-learning_rate, vr_gradient)

                # Update running iterate mean:
                param_state['running_x'].mul_(self.running_interp).add_(1-self.running_interp, p.data)

        # track number of minibatches seen
        self.batches_processed += 1

        dist = math.sqrt(dist_sq_acum)
        grad_dist = math.sqrt(grad_dist_sq_acum)

        self.inrun_iterate_distances.append(dist)
        self.inrun_grad_distances.append(grad_dist)

        return loss
