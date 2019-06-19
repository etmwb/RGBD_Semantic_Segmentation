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

from ..functions.depth_deform_conv_func import DepthDeformConvFunction

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

class Temp(DepthDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1, fr_fa=False):
        super(Temp, self).__init__(in_channels, out_channels,
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
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.conv_offset.lr_mult = lr_mult 
        self.conv_mask.lr_mult = lr_mult 
        # self.gamma.lr_mult = lr_mult 
        # self.fr_fa = fr_fa
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()
        self.conv_offset.weight.requires_grad = False
        self.conv_offset.bias.requires_grad = False
        self.conv_mask.weight.requires_grad = False
        self.conv_mask.bias.requires_grad = False

    def forward(self, input, depth):
        depth_offset = self.doffset(depth)
        depth_offset = depth_offset.detach()

        offset = self.conv_offset(input) 
        offset = offset + depth_offset.float() 
        depth_offset = torch.zeros_like(depth_offset)
        # print(offset)
        # print(np.histogram(offset.cpu().numpy(), bins=10), '\n')
        mask = 2*torch.sigmoid(self.conv_mask(input))
        depth = torch.zeros_like(depth)
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

    def forward(self, depth): 
        search_dilation = self.dilation[0] // 2 

        b, _, h, w = depth.size() 
        outH = (h + (2*self.padding[0]) - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1 
        outW = (w + (2*self.padding[1]) - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1 

        if search_dilation == 0: return torch.zeros(b, 18, outH, outW).int().cuda()

        sample = F.unfold(depth, kernel_size=3, dilation=search_dilation, padding=search_dilation, stride=1).view(b, 9, h, w) 
        sample = F.unfold(sample, kernel_size=self.kernel_size, dilation=self.dilation, 
                                padding=self.padding, stride=self.stride)
        center = F.unfold(depth, kernel_size=self.kernel_size, dilation=self.dilation, 
                                padding=self.padding, stride=self.stride)[:,4:5,:]

        sample = torch.abs(sample - center) 
        sample = sample.view(b, 9, 9, outH, outW)

        row_idx, col_idx = (0,1,2,6,7,8), (0,2,3,5,6,8)
        sample[:, col_idx, 1, :, :] = float('inf')
        sample[:, row_idx, 3, :, :] = float('inf')
        sample[:, row_idx, 5, :, :] = float('inf')
        sample[:, col_idx, 7, :, :] = float('inf')

        depth_idx = torch.min(sample, dim=1)[1] 
        depth_offset_h = (depth_idx // self.kernel_size[0] - 1) * search_dilation
        depth_offset_w = (depth_idx % self.kernel_size[1] - 1) * search_dilation
        depth_offset = torch.cat((depth_offset_h, depth_offset_w), dim=1).int()
        depth_offset = torch.zeros(b, 18, outH, outW).int().cuda()

        return depth_offset

class EliminateConv(nn.Module): 

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, bias=False):
        super(EliminateConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.use_bias = bias 

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
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

    def forward(self, input, depth):
        depth_unfold = F.unfold(depth, kernel_size=self.kernel_size, stride=self.stride, 
                                padding=self.padding, dilation=self.dilation) 
        depth_center = depth_unfold[:,4:5,:]
        eliminate_idx = torch.max(torch.abs(depth_unfold-depth_center), dim=1, keepdim=True)[1]
        eliminate_mask = torch.ones_like(depth_unfold)
        eliminate_mask.scatter_(1, eliminate_idx, 0.)

        b, _, h, w = input.size() 
        outH = (h + (2*self.padding[0]) - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1 
        outW = (w + (2*self.padding[1]) - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1 
        input_unfold = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, 
                                padding=self.padding, dilation=self.dilation)
        input_unfold = input_unfold.view(b, self.in_channels, 9, -1)
        eliminate_mask = eliminate_mask[:, None]
        input_unfold = (input_unfold * eliminate_mask).view(b, self.in_channels*9, -1)

        kernel_flat = self.weight.data.view(self.out_channels, -1)
        output = kernel_flat @ input_unfold
        output = output.view(b, self.out_channels, outH, outW)

        bias = self.bias.data.view(1, self.out_channels, 1, 1)
        output = output + bias 
        return output
