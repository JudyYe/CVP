# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import box_utils
from . import layout as layout_utils



def refine_module(din, dout, norm, act, num=2):
    layers = []
    layers.append(nn.Conv2d(din, dout, kernel_size=3, padding=1))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(dout))
    elif norm == 'none':
        pass
    else:
        raise NotImplementedError
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'leakyrelu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    for i in range(num -1):
        layers.append(nn.Conv2d(dout, dout, kernel_size=3, padding=1))
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(dout))
        elif norm == 'none':
            pass
        else:
            raise NotImplementedError
        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Comb(nn.Module):
    def __init__(self, feat_dim_list, dims, radius, normalization='instance', activation='leakyrelu'):
        """
        :param dims: channel dimensions
        """
        super().__init__()
        pose_dim, bbox_dim, embd_dim = feat_dim_list
        self.pose_dim = pose_dim
        self.bbox_dim = bbox_dim
        self.embd_dim = embd_dim
        self.bbox_w = radius

        layers = []
        for i in range(len(dims)):
            if i == 0:
                input_dim = pose_dim
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.obj_net = nn.Sequential(*layers)

        self.mask_net = nn.Sequential(*[
            nn.Conv2d(dims[-1], 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        ])

        self.bg_net = nn.ModuleList()
        # 0: 7*7*512 ;1: 14*14*x; 2:28*28*d...
        for i in range(len(dims)):
            if i == 0:
                layers = []
                input_dim = embd_dim[i]
            elif i < len(embd_dim):
                self.bg_net.append(nn.Sequential(*layers))
                layers = []
                input_dim = dims[i - 1] + embd_dim[i]
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.bg_net.append(nn.Sequential(*layers))

        layers = []
        output_dim = dims[-1]
        for i in range(2):
            input_dim = output_dim
            if input_dim <= 8:
                break
            output_dim = output_dim // 2 if input_dim > 8 else 8
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if input_dim > 8:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.extend([
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_dim, 3, kernel_size=1, padding=0)
        ])
        self.merge_net = nn.Sequential(*layers)

    def forward(self, pose, bbox_c, target_size, bg_feat):
        """
        :param pose: (Dt, V, o, D, H, W)
        :param bbox: (Dt, V, o, D)
        :param target_size:
        :param bg_feat (V, 1, D, H, W)
        :return: (Dt, V, 1, C, H, W)
        """
        # Figure out size of input
        # onput_H, onput_W = target_size
        dt, V, O, _, H, W = pose.size()
        layout = pose.view(dt * V * O, -1, H, W)

        obj_feat = self.obj_net(layout)
        obj_mask = self.mask_net(obj_feat)
        d, h, w = obj_feat.size()[-3:]
        obj_feat = obj_feat.view(dt * V, O, d, h, w)
        obj_mask = obj_mask.view(dt * V, O, 1, h, w)

        if bbox_c.size(-1) == 2:
            bbox = box_utils.xy_to_xyxy(bbox_c.view(dt * V * O, -1), self.bbox_w).view(dt * V, O, 4)
        elif bbox_c.size(-1) == 4:
            bbox = box_utils.centers_to_extents(bbox_c.view(dt * V * O, -1)).view(dt * V, O, 4)

        bg_feat = self._forward_bg(bg_feat, dt, V)  # (V, D, H, W)
        fb_feat, mask = layout_utils.mask_splat_to_bg(bbox, obj_mask, obj_feat, bg_feat)
        output = self.merge_net(fb_feat)


        output = output.view(dt, V, 1, 3, target_size[0], target_size[1])
        mask = mask.view(dt, V, O+1, 1, mask.size(-2), mask.size(-1))
        rtn = {
            'recon': output,
            'maskl': mask,
            # 'maskr': obj_mask
        }
        return rtn

    def _forward_bg(self, bg_feat, dt, V):
        """
        :param bg_feat: (N, Di, h, w)
        :param mask: (N, 1, H, W)
        :return: (N, Do, Hm, Wm)
        """
        blob_out = self.expand_bg(bg_feat[0], dt, V)
        for i in range(len(self.bg_net)):
            if i == 0:
                blob_in = blob_out
            elif i < len(bg_feat):
                blob_in = torch.cat([blob_out, self.expand_bg(bg_feat[i], dt, V)], dim=1)
            else:
                blob_in = blob_out
            blob_out = self.bg_net[i](blob_in)
        return blob_out

    def expand_bg(self, bg_feat, dt, V):
        bg_feat = bg_feat.view(V, bg_feat.size(-3), bg_feat.size(-2), bg_feat.size(-1))
        return bg_feat


class NoFactor(nn.Module):
    def __init__(self, feat_dim_list, dims, radius, normalization='instance', activation='leakyrelu'):
        """
        :param dims: channel dimensions
        """
        super().__init__()
        pose_dim, bbox_dim, embd_dim = feat_dim_list
        self.pose_dim = pose_dim
        self.bbox_dim = bbox_dim
        self.embd_dim = embd_dim
        self.bbox_w = radius

        layers = []
        for i in range(len(dims)):
            if i == 0:
                input_dim = pose_dim
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.extend(add_layer_to(8, dims[-1], normalization, activation))
        self.net = nn.Sequential(*layers)

        self.bg_net = nn.ModuleList()
        # 0: 7*7*512 ;1: 14*14*x; 2:28*28*d...
        for i in range(len(dims)):
            layers = []
            if i == 0:
                input_dim = embd_dim[i]
            elif i < len(embd_dim):
                input_dim = dims[i - 1] + embd_dim[i]
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.bg_net.append(nn.Sequential(*layers))
        self.bg_net.append(nn.Sequential(*add_layer_to(8, dims[-1], normalization, activation)))

    def forward(self, pose, bbox_c, target_size, bg_feat):
        """
        :param pose: (Dt, V, o, D, H, W)
        :param bbox: (Dt, V, o, D)
        :param target_size:
        :param bg_feat (V, 1, D, H, W)
        :return: (Dt, V, 1, C, H, W)
        """
        # Figure out size of input
        # onput_H, onput_W = target_size
        dt, V, O, _, H, W = pose.size()
        layout = pose.view(dt * V * O, -1, H, W)

        color = self.net(layout)
        color = color.view(dt * V, O, 3, target_size[0], target_size[1])

        bg = self._forward_bg(bg_feat, V)

        box_o = bbox_c.size(-2)
        if bbox_c.size(-1) == 2:
            bbox = box_utils.xy_to_xyxy(bbox_c.view(dt * V * box_o, 2), self.bbox_w).view(dt * V, box_o, 4)
        elif bbox_c.size(-1) == 4:
            bbox = box_utils.centers_to_extents(bbox_c.view(dt * V * box_o, 4)).view(dt * V, box_o, 4)

        layout, mask = layout_utils.bbox_to_bg(bbox, color, bg, target_size)
        output = layout.view(dt, V, 1, 3, target_size[0], target_size[1])
        # color = color.view(dt, V, O, 3, target_size[0], target_size[1])
        mask = mask.view(dt, V, 2, 1, target_size[0], target_size[1])

        rtn = {
            'recon': output,
            'maskl': mask,
        }
        return rtn

    def _forward_bg(self, bg_feat, V):
        """.view(V, -1, bg_feat.size(-2), bg_feat.size(-1)) """
        blob_in = bg_feat[0].view(V, -1, bg_feat[0].size(-1), bg_feat[0].size(-1))
        for i in range(len(self.bg_net)):
            if i == 0:
                pass
            elif i < len(bg_feat):
                blob_in = torch.cat([blob_out, bg_feat[i].view(V, -1, bg_feat[i].size(-1), bg_feat[i].size(-1))],
                                    dim=1)
            else:
                blob_in = blob_out
            blob_out = self.bg_net[i](blob_in)
        return blob_out


class Pix(nn.Module):
    def __init__(self, feat_dim_list, dims, radius, normalization='instance', activation='leakyrelu'):
        super().__init__()

        pose_dim, bbox_dim, embd_dim = feat_dim_list
        self.pose_dim = pose_dim
        self.bbox_dim = bbox_dim
        self.embd_dim = embd_dim
        self.bbox_w = radius

        self.mask_net = nn.Sequential(*[
            nn.Conv2d(dims[-1], 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        ])

        self.bg_net = nn.ModuleList()
        # 0: 7*7*512 ;1: 14*14*x; 2:28*28*d...
        for i in range(len(dims)):
            if i == 0:
                layers = []
                input_dim = embd_dim[i]
            elif i < len(embd_dim):
                self.bg_net.append(nn.Sequential(*layers))
                layers = []
                input_dim = dims[i - 1] + embd_dim[i]
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.bg_net.append(nn.Sequential(*layers))

        layers = []
        output_dim = dims[-1]
        for i in range(2):
            input_dim = output_dim
            if input_dim <= 8:
                break
            output_dim = output_dim // 2 if input_dim > 8 else 8
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if input_dim > 8:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.extend([
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_dim, 3, kernel_size=1, padding=0)
        ])
        self.merge_net = nn.Sequential(*layers)

        layers = []
        for i in range(len(dims)):
            if i == 0:
                input_dim = pose_dim
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.obj_net = nn.Sequential(*layers)

        self.color_head = nn.Sequential(*[
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
        ])

    def forward(self, pose, bbox_c, target_size, bg_feat):
        """merge_net(bg_net(bg)), mask with obj(obj_net)"""
        dt, V, O, _, H, W = pose.size()
        layout = pose.view(dt * V * O, -1, H, W)

        obj_feat = self.obj_net(layout)

        obj_mask = self.mask_net(obj_feat)
        obj_color = self.color_head(obj_feat)

        d, h, w = obj_color.size()[-3:]
        obj_color = obj_color.view(dt * V, O, d, h, w)
        obj_mask = obj_mask.view(dt * V, O, 1, h, w)

        if bbox_c.size(-1) == 2:
            bbox = box_utils.xy_to_xyxy(bbox_c.view(dt * V * O, -1), self.bbox_w).view(dt * V, O, 4)
        elif bbox_c.size(-1) == 4:
            bbox = box_utils.centers_to_extents(bbox_c.view(dt * V * O, -1)).view(dt * V, O, 4)

        bg_feat = self._forward_bg(bg_feat, dt, V)  # (V, D, H, W)
        back_color = self.merge_net(bg_feat)

        # bgfeat torch.Size([8, 32, 56, 56]) fgfeat torch.Size([8, 11, 32, 16, 16])
        fb_feat, mask = layout_utils.mask_splat_to_bg(bbox, obj_mask, obj_color, back_color)
        output = fb_feat

        output = output.view(dt, V, 1, 3, target_size[0], target_size[1])
        mask = mask.view(dt, V, O+1, 1, mask.size(-2), mask.size(-1))
        # obj_mask = obj_mask.view(dt, V, O, 1, obj_mask.size(-2), obj_mask.size(-1))
        rtn = {
            'recon': output,
            'maskl': mask,
            # 'maskr': obj_mask
        }
        return rtn

    def _forward_bg(self, bg_feat, dt, V):
        """
        :param bg_feat: (N, Di, h, w)
        :param mask: (N, 1, H, W)
        :return: (N, Do, Hm, Wm)
        """
        blob_out = self.expand_bg(bg_feat[0], dt, V)
        for i in range(len(self.bg_net)):
            if i == 0:
                blob_in = blob_out
            elif i < len(bg_feat):
                blob_in = torch.cat([blob_out, self.expand_bg(bg_feat[i], dt, V)], dim=1)
            else:
                blob_in = blob_out
            blob_out = self.bg_net[i](blob_in)
        return blob_out

    def expand_bg(self, bg_feat, dt, V):
        bg_feat = bg_feat.view(V, bg_feat.size(-3), bg_feat.size(-2), bg_feat.size(-1))
        return bg_feat


def add_layer_to(last, output_dim, normalization, activation, rgb=3):
    layers = []
    for i in range(2):
        input_dim = output_dim
        if input_dim <= last:
            break
        output_dim = output_dim // 2 if input_dim > last else last
        layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
        if input_dim > 8:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
    layers.extend([
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(output_dim, rgb, kernel_size=1, padding=0)])
    return layers




DecoderFactory = {
    'comb': Comb,
    'cPix': Pix,
    'noFactor': NoFactor,
}
