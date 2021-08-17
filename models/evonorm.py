#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:54:21 2021

@author: saketi
reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/evo_norm.py

"""
"""EvoNormB0 (Batched) and EvoNormS0 (Sample) in PyTorch
An attempt at getting decent performing EvoNorms running in PyTorch.
While currently faster than other impl, still quite a ways off the built-in BN
in terms of memory usage and throughput (roughly 5x mem, 1/2 - 1/3x speed).
Still very much a WIP, fiddling with buffer usage, in-place/jit optimizations, and layouts.
Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
import torch.nn as nn
import math


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum  = momentum
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self.running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) + self.running_var * (1 - self.momentum))
        else:
            var = self.running_var

        if self.apply_act:
            v = self.v.to(dtype=x_type)
            d = x * v + (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-5, drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v     = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = x * (x * self.v).sigmoid() # n = x*sigmoid(x*v)
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()  # x = n/var(x) groupwise instance variance
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias # gamma*x+beta



class RangeEN_full(nn.Module):
    def __init__(self, num_features, chunks=16, groups=8, apply_act=True, eps=1e-5):
        super(RangeEN_full, self).__init__()
        self.num_chunks = chunks
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v     = nn.Parameter(torch.ones(param_shape), requires_grad=True)

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        y = x.reshape(B, self.groups, self.num_chunks, -1)
        mean_max = y.max(-1)[0].mean(-1)  # C
        mean_min = y.min(-1)[0].mean(-1)  # C
        scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5) / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)
        #scale.view(1, scale.size(0), 1, 1)
        if self.apply_act:
            n = x * (x * self.v).sigmoid() # n = x*sigmoid(x*v)
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / scale.view(scale.size(0), scale.size(1), 1)  # x = n/var(x) groupwise instance variance
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias # gamma*x+beta


#layer = EvoNormSample2d(16)
#layer = nn.BatchNorm2d(16)
#x     = torch.rand(32,16,4,4)
#for p in layer.parameters():
#    print(p.shape)
#out   = layer(x)