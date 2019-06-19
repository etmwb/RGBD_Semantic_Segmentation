#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
import numpy as np
from torch import nn
from torch.nn import init
import torch.nn.functional as F 
from torch.nn.modules.utils import _pair
from nn import Mask_Module

from ..functions.depth_deform_conv_func import DepthDeformConvFunction
from .deform_conv import DeformConvOffsetPack

class DepthDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DepthDeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias 

        self.doffset = DepthOffset(self.kernel_size, self.stride, self.padding, self.dilation)

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, depth, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        depth_offset = self.doffset(depth)
        depth_offset = depth_offset.detach()
        return DepthDeformConvFunction.apply(input, depth, depth_offset, offset, mask,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DepthDeformConv = DepthDeformConvFunction.apply

class DepthDeformConvPack(DepthDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1, fr_fa=False):
        super(DepthDeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        self.doffset = DepthOffset(self.kernel_size, self.stride, self.padding, self.dilation)

        out_channels = self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                        2*out_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)
        self.conv_mask = nn.Conv2d(self.in_channels,
                                        out_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)
        # self.conv_mask = Mask_Module(self.in_channels,
        #                              kernel_size=self.kernel_size,
        #                              stride=self.stride,
        #                              padding=self.padding,
        #                              dilation=self.dilation)
        self.conv_offset.lr_mult = lr_mult 
        self.conv_mask.lr_mult = lr_mult * 0.1
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()
        # self.conv_mask.weight.requires_grad = False
        # self.conv_mask.bias.requires_grad = False
        # self.conv_offset.weight.requires_grad = False
        # self.conv_offset.bias.requires_grad = False

    def forward(self, input, depth):
        depth_offset = self.doffset(depth)
        depth_offset = depth_offset.detach()

        offset = self.conv_offset(input)
        # offset = offset * depth_offset.float() 
        depth_offset = torch.zeros_like(depth_offset) 
        # print(offset)
        # print(np.histogram(offset.cpu().numpy(), bins=10), '\n')
        mask = 2*torch.sigmoid(self.conv_mask(input))
        # mask = self.conv_mask(input)
        return DepthDeformConvFunction.apply(input, depth, depth_offset, offset, mask, 
                                                self.weight, 
                                                self.bias, 
                                                self.stride, 
                                                self.padding, 
                                                self.dilation, 
                                                self.groups,
                                                self.deformable_groups,
                                                self.im2col_step)


class DepthOffset(nn.Module): 
    def __init__(self, kernel_size, stride, padding, dilation=1): 
        super(DepthOffset, self).__init__()
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.avg_minus = Avg_Pool2d(kernel_size=self.kernel_size, dilation=tuple(map(lambda x: x-1, self.dilation)), 
                                    padding=tuple(map(lambda x: x-1, self.padding)), stride=self.stride)
        self.avg_plus = Avg_Pool2d(kernel_size=self.kernel_size, dilation=tuple(map(lambda x: x+1, self.dilation)), 
                                    padding=tuple(map(lambda x: x+1, self.padding)), stride=self.stride)
        self.avg_ori = Avg_Pool2d(kernel_size=self.kernel_size, dilation=self.dilation, 
                                    padding=self.padding, stride=self.stride)

    def forward(self, depth): 

        b, _, h, w = depth.size() 
        outH = (h + (2*self.padding[0]) - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1 
        outW = (w + (2*self.padding[1]) - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1 
        h_offset = torch.Tensor([-1,-1,-1,0,0,0,1,1,1]).long().cuda().view(1, 9, 1, 1)
        w_offset = torch.Tensor([-1,0,1,-1,0,1,-1,0,1]).long().cuda().view(1, 9, 1, 1)

        sample_ori = self.avg_ori(depth)
        sample_plus = self.avg_plus(depth)
        sample_center = F.unfold(depth, kernel_size=self.kernel_size, dilation=self.dilation, 
                                padding=self.padding, stride=self.stride)[:, 4:5, :].view(b, 1, outH, outW)
        sample_ori = torch.abs(sample_ori - sample_center)
        sample_plus = torch.abs(sample_plus - sample_center)
        if self.dilation != (1, 1): 
            sample_minus = self.avg_minus(depth)
            sample_minus = torch.abs(sample_minus - sample_center)
            sample = torch.cat([sample_minus, sample_ori, sample_plus], dim=1)
            sample_idx = torch.min(sample, dim=1, keepdim=True)[1] - 1
            depth_offset_h = (sample_idx * h_offset)
            depth_offset_w = (sample_idx * w_offset)

        else:
            sample = torch.cat((sample_ori, sample_plus), dim=1)
            sample_idx = torch.min(sample, dim=1, keepdim=True)[1]
            depth_offset_h = (sample_idx * h_offset)
            depth_offset_w = (sample_idx * w_offset)

        depth_offset = torch.cat((depth_offset_h, depth_offset_w), dim=1).int()
        # depth_offset = torch.zeros(b, 18, outH, outW).int().cuda()

        return depth_offset

class Avg_Pool2d(nn.Module): 

    def __init__(self, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation

    def forward(self, input): 
        b, c, h, w = input.size() 
        outH = (h + (2*self.padding[0]) - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1 
        outW = (w + (2*self.padding[1]) - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1 

        input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, 
                                padding=self.padding, stride=self.stride).view(b, c, self.kernel_size[0]*self.kernel_size[1], outH, outW)
        input_mean = torch.mean(input_unfold, dim=2)

        return input_mean