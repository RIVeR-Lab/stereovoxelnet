# Copyright 2021 Faranak Shamsafar
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from functools import reduce 
from spconv.pytorch import SparseConvTensor
import spconv.pytorch as spconv
from typing import List

###############################################################################
""" Fundamental Building Blocks """
###############################################################################


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
        nn.BatchNorm3d(out_channels)
    )


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False)
            )
    else:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )


class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class MobileV2_Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, 4]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileV2_Residual_3D(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio):
        super(MobileV2_Residual_3D, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expanse_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

###############################################################################
""" Feature Extraction """
###############################################################################
class feature_extraction(nn.Module):
    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()

        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        else:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 32, 2, self.expanse_ratio),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                           MobileV2_Residual(32, 32, 1, self.expanse_ratio)
                                           )

        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 32, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 64, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 64, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        feature_volume = torch.cat((l2, l3, l4), dim=1)

        return feature_volume

###############################################################################
""" Cost Volume Related Functions """
###############################################################################


def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
    interwoven_features[:,::2,:,:] = refimg_fea
    interwoven_features[:,1::2,:,:] = targetimg_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

###############################################################################
""" Disparity Regression Function """
###############################################################################


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

###############################################################################
""" Loss Function """
###############################################################################

def calc_IoU(pred, gt):
    intersect = pred*gt 
    total = pred+gt  
    union = total-intersect

    return (intersect.sum() + 1.0) / (union.sum() + 1.0)

def IoU_loss(pred, gt):
    return 1-calc_IoU(pred, gt)

def sparse_loss(*tens: SparseConvTensor):
    """reuse torch.sparse. the internal is sort + unique 
    """
    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: List[torch.Tensor] = []
    first = tens[0]
    res_shape = [first.batch_size, *first.spatial_shape, first.features.shape[1]]

    for i, ten in enumerate(tens):
        assert ten.spatial_shape == tens[0].spatial_shape
        assert ten.batch_size == tens[0].batch_size
        assert ten.features.shape[1] == tens[0].features.shape[1]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        ten_ths.append(torch.sparse_coo_tensor(ten.indices.T, ten.features, res_shape, requires_grad=True))
    
    c_th_intersect = reduce(lambda x, y: x * y, ten_ths).coalesce()
    c_th_total = reduce(lambda x, y: x + y, ten_ths).coalesce()
    c_th_union = c_th_total - c_th_intersect
    iou = (torch.sparse.sum(c_th_intersect) + 1.0) / (torch.sparse.sum(c_th_union) + 1.0)
    return 1-iou 

def model_loss(voxel_ests, voxel_gt, weighted_loss=None):
    voxel_ests = voxel_ests[0]
    # sum of loss of every level
    # from OGN https://openaccess.thecvf.com/content_ICCV_2017/papers/Tatarchenko_Octree_Generating_Networks_ICCV_2017_paper.pdf

    # weighted loss
    # from Quadtree

    # By default
    if weighted_loss:
        weight = [0.30,0.27,0.23,0.20]
    else:
        weight = [0.25,0.25,0.25,0.25]
    all_losses = []

    if isinstance(voxel_ests, torch.Tensor):
        # straight network
        all_losses.append(IoU_loss(voxel_ests, voxel_gt[-1]))
        return sum(all_losses), 1-sum(all_losses)

    if isinstance(voxel_ests[-1], SparseConvTensor):
        # calculate mix of iou loss and sparse loss
        for idx, voxel_est in enumerate(voxel_ests):
            loss = None
            if idx == 0:
                # dense loss for first output
                loss = IoU_loss(voxel_est, voxel_gt[idx])
            else:
                # calculate sparse loss
                sparse_label1 = torch.unsqueeze(voxel_gt[idx].clone(),1).permute(0,2,3,4,1)
                sparse_label1 = spconv.SparseConvTensor.from_dense(sparse_label1)    
                loss = sparse_loss(voxel_est, sparse_label1)
            all_losses.append(weight[idx] * loss)
        return sum(all_losses), 1-torch.mean(torch.Tensor(all_losses))
        
    else:
        for idx, voxel_est in enumerate(voxel_ests):
            all_losses.append(weight[idx] * IoU_loss(voxel_est, voxel_gt[idx]))
        return sum(all_losses), 1-sum(all_losses)