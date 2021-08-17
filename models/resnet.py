#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:16:56 2021

@author: saketi
"""

import torch.nn as nn
import torchvision.transforms as transforms
import math
from .evonorm import EvoNormSample2d as evonorm_s0

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
def normalization(planes, groups=2, norm_type='batchnorm'):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(planes)
    if norm_type == 'groupnorm':
        return nn.GroupNorm(groups, planes)
    if norm_type == 'evonorm':
        return evonorm_s0(planes)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, RangeBN_full):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_type, groups=2, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.norm_type = norm_type
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normalization(planes, groups, norm_type)
        if self.norm_type!='evonorm':
            self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normalization(planes, groups, norm_type)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.norm_type!='evonorm':
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.norm_type!='evonorm':
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, norm_type='batchnorm', groups=None, stride=1):
        downsample     = None
        self.norm_type = norm_type
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normalization(planes * block.expansion, groups, norm_type),
            )

        layers = []
       
        layers.append(block(self.inplanes, planes, norm_type, groups, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type, groups,))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.norm_type!='evonorm':
            x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)
    

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18, groups=None, norm_type='batchnorm'):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        self.norm_type = norm_type
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
  
        self.bn1 = normalization(16, groups, norm_type)
        if self.norm_type!='evonorm':
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n, norm_type, groups)
        self.layer2 = self._make_layer(block, 32, n, norm_type, groups, stride=2)
        self.layer3 = self._make_layer(block, 64, n, norm_type, groups, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        init_model(self)
      

def resnet(**kwargs):
    num_classes, depth, dataset, norm_type, groups = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'norm_type', 'groups'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth       = depth or 56
        return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth, norm_type=norm_type, groups=groups)