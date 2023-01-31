# created by Hongyu Li at 20230130 18:55.
# 
# https://lhy.xyz

from __future__ import print_function
import math
import torch.nn as nn
import torch.utils.data
from torch import reshape
import torch.nn.functional as F
from .submodule import feature_extraction, convbn, interweave_tensors, groupwise_correlation
from spconv.pytorch import SparseConvTensor
import spconv.pytorch as spconv
from functools import reduce 
from typing import List

def calc_IoU(pred, gt):
    intersect = pred*gt 
    total = pred+gt  
    union = total-intersect

    return (intersect.sum() + 1.0) / (union.sum() + 1.0)

def IoU_loss(pred, gt):
    return 1-calc_IoU(pred, gt)

def sparse_loss(*tens: SparseConvTensor):
    """
    calculate the IoU loss of the sparse matrix
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

def create_sparse_tensor(x: torch.Tensor):
    """create sparse tensor fron channel last dense tensor by to_sparse
    x must be NHWC tensor, channel last
    """
    x_sp = x.to_sparse(x.ndim - 1).coalesce()
    spatial_shape = list(x_sp.shape[1:-1])
    batch_size = x_sp.shape[0]
    indices_th = x_sp.indices().permute(1, 0).contiguous().int()
    features_th = x_sp.values()
    return spconv.SparseConvTensor(features_th, indices_th, spatial_shape, batch_size)


class UNet(nn.Module):
    def __init__(self, cost_vol_type) -> None:
        super(UNet, self).__init__()
        # 48x128x240 => 64x64x128
        if cost_vol_type == "full":
            self.conv1 = nn.Sequential(nn.Conv2d(48, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                    nn.ReLU(inplace=True))
        elif cost_vol_type == "voxel" or cost_vol_type == "eveneven":
            self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                    nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                    nn.ReLU(inplace=True))

        # 64x64x128 => 128x16x32
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                   nn.ReLU(inplace=True))

        # 128x16x32 => 256x4x8
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                   nn.ReLU(inplace=True))

        self.linear1 = nn.Sequential(
            nn.Linear(256*3*7, 512), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True))

        # 256x1x1x1 => 256x2x2x2
        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(128, 64, kernel_size=(5, 5, 5), stride=(2, 2, 2), padding=0, bias=False),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(64, 64, kernel_size=2, bias=False),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=(5, 5, 5), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(),
                                     nn.Conv3d(32, 32, kernel_size=2, bias=False),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU())

        self.deconv2_out = nn.Sequential(nn.Conv3d(32, 1, kernel_size=1, bias=False),
                                        nn.Sigmoid())

        self.deconv3 = spconv.SparseSequential(spconv.SparseConvTranspose3d(32, 16, kernel_size=6, stride=2, padding=2, bias=False),
                                     nn.BatchNorm1d(16),
                                     nn.ReLU(),
                                     spconv.SubMConv3d(16,16,kernel_size=3, bias=False),
                                     nn.BatchNorm1d(16),
                                     nn.ReLU())
        self.deconv3_out = spconv.SparseSequential(spconv.SparseConv3d(16, 1, kernel_size=1, bias=False),
                                        nn.Sigmoid())
        self.deconv4 = spconv.SparseSequential(spconv.SparseConvTranspose3d(16, 8, kernel_size=6, stride=2, padding=2, bias=False),
                                     nn.BatchNorm1d(8),
                                     nn.ReLU(),
                                     spconv.SubMConv3d(8,8,kernel_size=3, bias=False),
                                     nn.BatchNorm1d(8),
                                     nn.ReLU())
        self.deconv4_out = spconv.SparseSequential(spconv.SubMConv3d(8, 1, kernel_size=1, bias=False),
                                        nn.Sigmoid())
        self.deconv5 = spconv.SparseSequential(spconv.SparseConvTranspose3d(8, 1, kernel_size=6, stride=2, padding=2),
                                     nn.Sigmoid())

    def forward(self, x, level=None, label=None):
        if self.training:
            level=None

        B, C, H, W = x.shape

        # encoding
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = reshape(conv3, (B, -1,))

        # latent
        linear1 = self.linear1(conv3)
        linear2 = self.linear2(linear1)
        latent = self.linear3(linear2)

        # decoding
        latent = reshape(latent, (B, 128, 1, 1, 1))

        deconv1 = self.deconv1(latent)
        deconv2 = self.deconv2(deconv1)

        if level is None or level=="2":
            out_2 = self.deconv2_out(deconv2)
            out_2 = torch.squeeze(out_2, 1)
            if level=="2":
                return out_2
        
        if label is None:
            # in prediction
            mask_2 = out_2.clone().detach()
            mask_2[mask_2 >= 0.5] = 1
            mask_2[mask_2 < 0.5] = 0
            mask_2 = torch.unsqueeze(mask_2,1).repeat(1,32,1,1,1).type(deconv2.dtype)
        else:
            # in training
            mask_2 = torch.unsqueeze(label[0],1).repeat(1,32,1,1,1).type(deconv2.dtype)
        
        deconv2 = deconv2 * mask_2
        deconv2 = deconv2.permute(0,2,3,4,1)
        deconv2 = create_sparse_tensor(deconv2)    
        deconv3 = self.deconv3(deconv2)

        sparse_out_3 = self.deconv3_out(deconv3)
        out_3 = sparse_out_3.dense()
        out_3 = torch.squeeze(out_3, 1)
        if level=="3":
            return out_3

        deconv3 = deconv3.dense()
        if label is None:
            # in prediction
            mask_3 = out_3.clone().detach()
            mask_3[mask_3 >= 0.5] = 1
            mask_3[mask_3 < 0.5] = 0
            mask_3 = torch.unsqueeze(mask_3,1).repeat(1,16,1,1,1).type(deconv3.dtype)

            # extreme case, avoid bug in sparse calculation
            if not torch.is_nonzero(mask_3.sum()):
                mask_3 += 1e-6
        else:
            # in training
            mask_3 = torch.unsqueeze(label[1],1).repeat(1,16,1,1,1).type(deconv3.dtype)
        
        deconv3 = deconv3 * mask_3
        deconv3 = deconv3.permute(0,2,3,4,1)
        deconv3 = create_sparse_tensor(deconv3)    
        deconv4 = self.deconv4(deconv3)

        sparse_out_4 = self.deconv4_out(deconv4)
        out_4 = sparse_out_4.dense()
        out_4 = torch.squeeze(out_4, 1)
        if level=="4":
            return out_4

        deconv4 = deconv4.dense()
        if label is None:
            # in prediction
            mask_4 = out_4.clone().detach()
            mask_4[mask_4 >= 0.5] = 1
            mask_4[mask_4 < 0.5] = 0
            mask_4 = torch.unsqueeze(mask_4,1).repeat(1,8,1,1,1).type(deconv4.dtype)

            # extreme case, avoid bug in sparse calculation
            if not torch.is_nonzero(mask_4.sum()):
                mask_4 += 1e-6
        else:
            # in training
            mask_4 = torch.unsqueeze(label[2],1).repeat(1,8,1,1,1).type(deconv4.dtype)

        deconv4 = deconv4 * mask_4
        deconv4 = deconv4.permute(0,2,3,4,1)
        deconv4 = create_sparse_tensor(deconv4)
        sparse_out_5 = self.deconv5(deconv4)
        out_5 = sparse_out_5.dense()
        out_5 = torch.squeeze(out_5, 1)
        if level=="5":
            return out_5

        if label:
            # know the ground truth
            weight = [0.30,0.27,0.23,0.20]
            all_losses = []
            voxel_ests = [out_2, sparse_out_3, sparse_out_4, sparse_out_5]
            for idx, voxel_est in enumerate(voxel_ests):
                loss = None
                if idx == 0:
                    # dense loss for first output
                    loss = IoU_loss(voxel_est, label[idx])
                else:
                    # calculate sparse loss
                    sparse_label = torch.unsqueeze(label[idx].clone(),1).permute(0,2,3,4,1)
                    sparse_label = create_sparse_tensor(sparse_label)    
                    loss = sparse_loss(voxel_est, sparse_label)
                all_losses.append(weight[idx] * loss)

            return [out_2, out_3, out_4, out_5], sum(all_losses), 1-sum(all_losses)
        else:
            # during inference
            return [out_2, out_3, out_4, out_5]


class Voxel2D(nn.Module):
    def __init__(self, maxdisp, cost_vol_type="even"):

        super(Voxel2D, self).__init__()

        self.maxdisp = maxdisp
        self.cost_vol_type = cost_vol_type

        self.num_groups = 1

        self.volume_size = 24

        self.hg_size = 64

        self.dres_expanse_ratio = 3

        self.feature_extraction = feature_extraction(add_relus=True)

        self.preconv11 = nn.Sequential(convbn(160, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1))

        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1], bias=False),
                                    nn.BatchNorm3d(8),
                                    nn.ReLU(),
                                    nn.Conv3d(8, 16, kernel_size=(4, 3, 3), stride=[
                                              4, 1, 1], padding=[0, 1, 1], bias=False),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 8, kernel_size=(2, 3, 3), stride=[
                                              2, 1, 1], padding=[0, 1, 1], bias=False),
                                    nn.BatchNorm3d(8),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(convbn(8, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.output_layer = nn.Sequential(nn.Conv2d(self.hg_size, self.hg_size, 1, 1, 0),
                                          nn.Sigmoid())

        self.encoder_decoder = UNet(self.cost_vol_type)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R, voxel_cost_vol=[0], level=None, label=None):
        features_L = self.feature_extraction(L)
        features_R = self.feature_extraction(R)

        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)

        B, C, H, W = featL.shape

        # default even = 24
        iter_size = self.volume_size
        
        if self.cost_vol_type == "full":
            # full disparity = 24x2 = 48
            iter_size = int(self.volume_size*2)
        elif self.cost_vol_type == "eveneven":
            # eveneven = 24/2 = 12
            iter_size = int(self.volume_size/2)
        elif self.cost_vol_type == "voxel":
            # voxel  = 17+1 = 18
            iter_size = len(voxel_cost_vol) + 1

        volume = featL.new_zeros([B, self.num_groups, iter_size, H, W])

        for i in range(iter_size):
            if i > 0:
                if self.cost_vol_type == "even":
                    j = 2*i
                elif self.cost_vol_type == "eveneven":
                    j = 4*i
                elif self.cost_vol_type == "full":
                    j = i
                elif self.cost_vol_type == "voxel":
                    j = int(voxel_cost_vol[i-1][0])

                x = interweave_tensors(featL[:, :, :, j:], featR[:, :, :, :-j])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, j:] = x
            else:
               
                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        out = self.encoder_decoder(volume, level, label)
        return [out]