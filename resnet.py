# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''ResNet in PyTorch.
from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
based on
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batchnorm=True, nonlinearity=None):
        super(PreActBlock, self).__init__()
        self.batchnorm = batchnorm

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.nonlinearity = nonlinearity

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.batchnorm:
            out = self.bn1(x)
        else:
            out = x
        out = self.nonlinearity(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.nonlinearity(out)
        out = self.conv2(out)
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batchnorm=True, nonlinearity=None):
        super(Bottleneck, self).__init__()
        self.batchnorm = batchnorm

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.nonlinearity = nonlinearity

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if batchnorm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.nonlinearity(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.nonlinearity(out)
        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)
        out += self.shortcut(x)
        out = self.nonlinearity(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batchnorm=True, nonlinearity=None):
        super(PreActBottleneck, self).__init__()
        self.batchnorm = batchnorm

        if batchnorm:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.nonlinearity = nonlinearity

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.batchnorm:
            out = self.bn1(x)
        else:
            out = x
        out = self.nonlinearity(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.conv2(self.nonlinearity(out))
        if self.batchnorm:
            out = self.bn3(out)
        out = self.conv3(self.nonlinearity(out))
        out += shortcut
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batchnorm=True, nonlinearity=None):
        super(ResNetImageNet, self).__init__()
        self.batchnorm = batchnorm
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.nonlinearity = nonlinearity

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride,
                batchnorm=self.batchnorm, nonlinearity=self.nonlinearity))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.nonlinearity(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetCifar(nn.Module):
    def __init__(self, block, blocks_per=3, num_classes=10,
                 batchnorm=True, nonlinearity=None, in_planes=16):
        super(ResNetCifar, self).__init__()
        self.batchnorm = batchnorm
        self.in_planes = in_planes # standard resnet is 16
        self.nonlinearity = nonlinearity

        self.conv1 = conv3x3(3, in_planes)
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, blocks_per, stride=1)
        self.layer2 = self._make_layer(block, 2*in_planes, blocks_per, stride=2)
        self.layer3 = self._make_layer(block, 4*in_planes, blocks_per, stride=2)
        self.linear = nn.Linear(4*in_planes*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride,
                batchnorm=self.batchnorm, nonlinearity=self.nonlinearity))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.nonlinearity(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #import pdb; pdb.set_trace() # torch.Size([64, 64, 8, 8]) good.
        outp = F.avg_pool2d(out, 8) # torch.Size([64, 64, 1, 1])
        outv = outp.view(outp.size(0), -1) # after torch.Size([64, 256])
        outl = self.linear(outv) # want 64x64?
        return outl


def ResNet18(**kwargs):
    return ResNetImageNet(PreActBlock, [2,2,2,2], **kwargs)

def ResNet50(**kwargs):
    return ResNetImageNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNetSmallWide(**kwargs):
    return ResNetCifar(PreActBlock, blocks_per=1, **kwargs)

def ResNetSmall(**kwargs):
    return ResNetCifar(PreActBlock, blocks_per=3, in_planes=8, **kwargs)

def ResNet(**kwargs):
    return ResNetCifar(PreActBlock, **kwargs)

# I get out-of-memory errors for these larger ones
def ResNet56(**kwargs):
    return ResNetCifar(PreActBlock, blocks_per=9, **kwargs)

def ResNet110(**kwargs):
    return ResNetCifar(PreActBlock, blocks_per=18, **kwargs)
