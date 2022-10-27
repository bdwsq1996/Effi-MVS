import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()

        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea


def homo_warping2(src_fea, src_proj, ref_proj, depth_values):
    "a homo_warping function to solve 'None' problem"
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()

        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:, 2:3, :, :] = torch.where(proj_xyz[:, 2:3, :, :] == 0, proj_xyz[:, 2:3, :, :] + 1e-8,
                                             proj_xyz[:, 2:3, :, :])
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        if proj_xyz[:, 2:3, :, :].mean() != proj_xyz[:, 2:3, :, :].mean():
            proj_xyz[:, 2:3, :, :] += 1e8
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    return warped_src_fea



class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class P_1to8_FeatureNet(nn.Module):
    def __init__(self, base_channels=8, out_channel=[32,16,8], stage_channel=True):
        super(P_1to8_FeatureNet, self).__init__()


        self.base_channels = base_channels


        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )
        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )
        if stage_channel:
            self.out1 = nn.Conv2d(base_channels * 8, out_channel[0], 1, bias=False)
        else:
            self.out1 = nn.Conv2d(base_channels * 8, out_channel[1], 1, bias=False)
        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        if stage_channel:
            self.out2 = nn.Conv2d(final_chs, out_channel[1], 3, padding=1, bias=False)
            self.out3 = nn.Conv2d(final_chs, out_channel[2], 3, padding=1, bias=False)
        else:
            self.out2 = nn.Conv2d(final_chs, out_channel[1], 3, padding=1, bias=False)
            self.out3 = nn.Conv2d(final_chs, out_channel[1], 3, padding=1, bias=False)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        intra_feat = conv3
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv2)
        out = self.out2(intra_feat)
        outputs["stage2"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv1)
        out = self.out3(intra_feat)
        outputs["stage3"] = out


        return outputs

class CostRegNet_small(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet_small, self).__init__()

        self.conv0 = Conv3d(in_channels, base_channels, padding=1)
        self.conv1 = Conv3d(base_channels, base_channels, padding=1)

        self.conv2 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv3 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv4 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv5 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv6 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv7 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)
        #
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        conv1 = self.conv1(self.conv0(x))
        conv3 = self.conv3(self.conv2(conv1))
        x = self.conv5(self.conv4(conv3))
        x = conv3 + self.conv6(x)
        x = conv1 + self.conv7(x)
        x = self.prob(x)
        return x

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def smooth_loss(inputs, depth_gt_ms, mask_ms, dloss, loss_rate=0.9):
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    loss_dict = {}
    loss_len = len(inputs)
    loss_rate = loss_rate
    stage_id = dloss
    for i, stage_inputs in enumerate(inputs):
        depth_est = stage_inputs
        depth_gt = depth_gt_ms["stage{}".format(stage_id[i])]
        mask = mask_ms["stage{}".format(stage_id[i])]
        mask = mask > 0.5
        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        loss_dict["l{}".format(i)] = depth_loss
        if i == 0:
            total_loss += 1.0 * depth_loss
        else:
            total_loss += (loss_rate ** (loss_len - i - 1)) * depth_loss
    return total_loss, loss_dict

def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth // 2 * depth_inteval_pixel).clamp(min = 1e-5)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth // 2 * depth_inteval_pixel).clamp(min = 1e-5, max = 1e5)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=1e-5)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel)#.clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples.clamp(min = 1e-5)


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples

if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # MVSDataset = find_dataset_def("colmap")
    # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    #                      128, interval_scale=1.06, max_h=1250, max_w=1024)

    MVSDataset = find_dataset_def("dtu_yao")
    num_depth = 48
    dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, num_depth, interval_scale=1.06 * 192 / num_depth)

    dataloader = DataLoader(dataset, batch_size=1)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

    warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)

    ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    cv2.imwrite('../tmp/ref.png', ref_img_np)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1] * 255

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
        cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))
