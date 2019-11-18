# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch.optim as optim
import torch_svrg
import recompute_svrg
import scsg

def optimizer(model, args):
    print("Using", args.method)

    if args.method == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay,
        momentum=args.momentum)
    elif args.method == "svrg":
        optimizer = torch_svrg.SVRG(model.parameters(), args=args, lr=args.lr,
        nbatches=args.nbatches,
        momentum=args.momentum, weight_decay=args.decay)
    elif args.method == "recompute_svrg":
        optimizer = recompute_svrg.RecomputeSVRG(model.parameters(), lr=args.lr,
        nbatches=args.nbatches, model=model, vr_bn_at_recalibration=args.vr_bn_at_recalibration,
        vr_from_epoch=args.vr_from_epoch,
        momentum=args.momentum, weight_decay=args.decay)
    elif args.method == "scsg":
        optimizer = scsg.SCSG(model.parameters(), args=args, lr=args.lr,
        nbatches=args.nbatches, model=model, vr_bn_at_recalibration=args.vr_bn_at_recalibration,
        vr_from_epoch=args.vr_from_epoch,
        momentum=args.momentum, weight_decay=args.decay)
    else:
        raise Exception("Optimizer not recognised:", args.method)

    return optimizer
