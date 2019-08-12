# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sg2im.layers import build_mlp, deconv3d, batchNorm5d
from sg2im.decoder import refine_module, add_layer_to
from sg2im.utils import vid_batch_to_cuda

class PoseFrameGan(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.nlayers = 5
        self.dnet = None
        self.gnet = FrameG(args.dec_dims)

    def fw_g(self, batch):
        if isinstance(batch, tuple) and len(batch) == 2:
            image, pose = batch
        else:
            image = batch['image'][0]
            pose = batch['pose']
        fake = self.gnet(image, pose)  #(dt, V, 1, C, H, W)

        pred = {}
        pred['fake'] = fake
        return pred

    def fw_d(self, batch, pred):
        # real_label = self.dnet(batch['image'])  #(V, 1?)
        # fake_label = self.dnet(pred['fake'].detach())
        #
        # pred['real_label'] = real_label
        # pred['fake_label'] = fake_label
        return


class FrameG(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.down = None
        self.up = nn.ModuleList()
        self.skip_layer = [7, 6, 5]
        self.defineUnetG(dims)

    def defineUnetG(self, dims):
        normalization = 'batch'
        activation = 'leakyrelu'
        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-2]
        modules[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        embed_dim = []

        for skip in self.skip_layer:
            embed_dim.append(modules[skip][1].bn2.num_features)
        self.down = nn.ModuleList(modules)

        for i in range(len(dims)):
            layers = []
            if i == 0:
                input_dim = embed_dim[i]
            elif i < len(embed_dim):
                input_dim = dims[i - 1] + embed_dim[i]
            else:
                input_dim = dims[i - 1]
            output_dim = dims[i]
            layers.extend(refine_module(input_dim, output_dim, normalization, activation, 1))
            if i < len(dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.up.append(nn.Sequential(*layers))
        self.up.append(nn.Sequential(*add_layer_to(8, dims[-1], normalization, activation)))

    def forward(self, bg, poses):
        dt, V, o, C, H, W = poses.size()
        bg = bg.view(V, C, H, W)
        poses = poses.view(dt, V, C, H, W)
        assert V == 1 and o == 1
        vid_inp = []
        v = 0
        for t in range(dt):
            one_step = torch.cat([bg[v], poses[t, v]], dim=0)
            vid_inp.append(one_step)
        vid_inp = torch.stack(vid_inp, dim=0)  # (dt * V, C,  H, W)
        assert vid_inp.size(0) == dt

        skip = self.down_forward(vid_inp)
        out = self.up_forward(skip)  # (dt * V, C, H, W)
        out = out.view(dt, V, o, C, H, W)
        return out

    def down_forward(self, vid):
        skip = []
        inp = vid
        for i in range(len(self.down)):
            inp = self.down[i](inp)
            skip.append(inp)
        return skip

    def up_forward(self, skip):
        nlayers = len(skip)
        for n in range(len(self.up)):
            back_n = nlayers - 1 - n
            if back_n in self.skip_layer:
                context = skip[back_n]
                if n == 0:
                    inp = context
                else:
                    inp = torch.cat([inp, context], dim=1)
            else:
                inp = inp
            inp = self.up[n](inp)
        return inp


class PoseVidGan(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nlayers = 5
        self.dnet = VidD(self.nlayers)
        self.gnet = VidG(self.nlayers)

    def forward(self, batch):
        """
        :param x: (V, C, H, W), (DT, V, C, H, W)
        :return:
        """
        batch = vid_batch_to_cuda(batch)
        real_label = self.dnet(batch['image'])  #(V, 1?)
        fake = self.gnet(batch['image'][0], batch['pose'])  #(dt, V, 1, C, H, W)
        fake_label = self.dnet(fake)

        pred = {}
        pred['real_label'] = real_label
        pred['fake_label'] = fake_label
        pred['fake'] = fake
        return pred

    def fw_g(self, batch):
        if isinstance(batch, tuple) and len(batch) == 2:
            image, pose = batch
        else:
            image = batch['image'][0]
            pose = batch['pose']
        fake = self.gnet(image, pose)  #(dt, V, 1, C, H, W)
        fake_label = self.dnet(fake)

        pred = {}
        pred['fake_label'] = fake_label
        pred['fake'] = fake
        return pred

    def fw_d(self, batch, pred):
        real_label = self.dnet(batch['image'])  #(V, 1?)
        fake_label = self.dnet(pred['fake'].detach())

        pred['real_label'] = real_label
        pred['fake_label'] = fake_label
        return pred


class VidD(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.nlayers = layers
        self.out_nc = 2
        self.model = self.defineD(layers)

    def defineD(self, nlayer=5, t=8):
        ngf = 128
        unit = []
        unit.append(nn.Conv3d(3, ngf, 4, 2, 1))
        unit.append(nn.LeakyReLU(0.2, True))
        t //= 2
        for i in range(1, nlayer - 1):
            if t <= 1:
                kt, st, pt = 1, 1, 0
            else:
                kt, st, pt = 4, 2, 1
            unit.append(nn.Conv3d(ngf, 2 * ngf, (kt, 4, 4), (st, 2, 2), (pt, 1, 1)))
            unit.append(nn.BatchNorm3d(2 * ngf))
            unit.append(nn.LeakyReLU(0.2, True))

            ngf *= 2
            t //= 2
        unit.append(nn.AdaptiveMaxPool3d((1, 4, 4)))
        unit.append(nn.Conv3d(ngf, self.out_nc, (1, 4, 4), 1, 0))
        unit.append(nn.Softmax(dim=1))
        netd = nn.Sequential(*unit)
        return netd

    def forward(self, x):
        """
        :param x: (dt, V, C, H, W)
        :return:
        """
        dt, V, o, C, H, W = x.size()
        # assert V == 1
        cur_inp = x.view(dt, C, H, W)
        cur_inp = cur_inp.permute(1, 0, 2, 3).unsqueeze(0)
        # skip = []
        # for n in range(self.nlayers - 1):
        #     cur_inp = self.model[n](cur_inp)
        #     skip.append(cur_inp)
        # label = self.model[self.nlayers - 1](cur_inp).view(V, self.out_nc)
        label = self.model(cur_inp)
        return label


class VidG(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.nlayers = layers

        self.down = None
        self.up = None
        self.defineUnetG(layers)

    def defineUnetG(self, nlayers, t=8):
        netd = nn.ModuleList()
        ngf = 32
        unit = []
        unit.append(nn.Conv3d(6, ngf, 4, 2, 1))
        netd.append(nn.Sequential(*unit))
        t //= 2
        time_kernel = []
        for i in range(1, nlayers):
            unit = []
            if t <= 1:
                kt, st, pt = 1, 1, 0
            else:
                kt, st, pt = 4, 2, 1
            unit.append(nn.LeakyReLU(0.2, True))
            unit.append(nn.Conv3d(ngf, 2 * ngf, (kt, 4, 4), (st, 2, 2), (pt, 1, 1)))
            unit.append(nn.BatchNorm3d(2 * ngf))
            netd.append(nn.Sequential(*unit))
            ngf *= 2
            t //= 2
            time_kernel.append((kt, st, pt))
        self.down = netd

        # (N, D, Dt, 4, 4)
        time_kernel.reverse()
        netg = nn.ModuleList()

        for i in range(0, nlayers - 1):
            unit = []
            kt, st, pt = time_kernel[i]
            unit.append(nn.LeakyReLU(0.2, True))
            if i == 0:
                unit.append(nn.ConvTranspose3d(ngf, ngf // 2, (kt, 4, 4), (st, 2, 2), (pt, 1, 1)))
            else:
                unit.append(nn.ConvTranspose3d(ngf * 2, ngf // 2, (kt, 4, 4), (st, 2, 2), (pt, 1, 1)))
            unit.append(batchNorm5d(ngf // 2))
            netg.append(nn.Sequential(*unit))
            ngf = ngf // 2
        unit = [nn.LeakyReLU(0.2, True),
                nn.ConvTranspose3d(ngf * 2, 3, 4, 2, 1)]
        netg.append(nn.Sequential(*unit))
        self.up = netg
        return

    def forward(self, bg, poses):
        """
        :param bg: (V, o, C, H, W)
        :param poses: (dt, V, 1, C, H, W)
        :return: (dt, V, 1, C, H, W)
        """
        dt, V, o, C, H, W = poses.size()
        bg = bg.view(V, C, H, W)
        poses = poses.view(dt, V, C, H, W)
        assert  V == 1 and o == 1
        vid_inp = []
        v = 0
        for t in range(dt):
            one_step = torch.cat([bg[v], poses[t, v]], dim=0)
            vid_inp.append(one_step)
        vid_inp = torch.stack(vid_inp, dim=1) # (C, dt, H, W)
        assert vid_inp.size(1) == dt
        vid_inp = vid_inp.unsqueeze(0)

        skip = self.down_forward(vid_inp)
        out = self.up_forward(skip)  # (V, C, dt, H, W)

        out = out.permute(2, 0, 1, 3, 4).unsqueeze(2)
        return out

    def down_forward(self, vid):
        skip = []
        inp = vid
        for i in range(len(self.down)):
            inp = self.down[i](inp)
            skip.append(inp)
        return skip

    def up_forward(self, skip):
        assert len(self.up) == len(skip) == self.nlayers, '%d %d' % (len(self.up), len(skip))
        for n in range(len(self.up)):
            context = skip[self.nlayers - 1 - n]
            if n == 0:
                inp = context
            else:
                inp = torch.cat([inp, context], dim=1)
            inp = self.up[n](inp)
        return inp


class PoseVaeNoFactor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.spatial = 1
        self.z_dim = 13 * args.bbox_dim
        self.args = args

        self.state_encoder = StateEncoder(512, self.z_dim)
        self.pose_decoder = PoseFCDecoder(self.z_dim * 2, self.z_dim)
        self.pose_encoder = PoseEncoder(self.z_dim * 2, self.z_dim)

    def forward(self, batch, test_sample=False):
        batch = vid_batch_to_cuda(batch)
        image = batch['image']  # (dt, V, 1, C, H, W)
        dt, V, _, C, H, W = image.size()
        bbox = batch['bbox'].view(dt, V, -1)    # (dt, V, O * 2)

        hidden_feat = self.state_encoder(image[0].view(V, C, H, W), bbox[0])  # (1, V, D)

        vel_bbox = bbox[1:] - bbox[0:-1]
        z, kl_loss, ori_z = self.pose_encoder((hidden_feat, bbox[1:], vel_bbox, test_sample))  # (dt - 1, V, De) -> (V, De)

        hidden_feat, cur_box = self.pose_decoder((hidden_feat, bbox[1:], z))  # (dt - 1, V, De)
        cur_box = cur_box.view(dt - 1, V, -1, 2)

        pred = {}
        pred['bbox'] = cur_box
        pred['kl_loss'] = kl_loss
        pred['orig_z'] = ori_z
        return pred


class PoseVaePos(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.spatial = 1
        self.z_dim = 13 * args.bbox_dim
        self.args = args

        self.state_encoder = StateEncoder(512, self.z_dim)
        self.pose_decoder = PoseDecoder(self.z_dim * 2, self.z_dim)
        self.pose_encoder = PoseEncoder(self.z_dim * 2, self.z_dim)

    def forward(self, batch, test_sample=False):
        batch = vid_batch_to_cuda(batch)
        image = batch['image']  # (dt, V, 1, C, H, W)
        dt, V, _, C, H, W = image.size()
        bbox = batch['bbox'].view(dt, V, -1)    # (dt, V, O * 2)

        hidden_feat = self.state_encoder(image[0].view(V, C, H, W), bbox[0])  # (1, V, D)

        vel_bbox = bbox[1:] - bbox[0:-1]
        z, kl_loss, ori_z = self.pose_encoder((hidden_feat, bbox[1:], vel_bbox, test_sample))  # (dt - 1, V, De) -> (V, De)

        hidden_feat, cur_box = self.pose_decoder((hidden_feat, bbox[1:], z))  # (dt - 1, V, De)
        cur_box = cur_box.view(dt - 1, V, -1, 2)

        pred = {}
        pred['bbox'] = cur_box
        pred['kl_loss'] = kl_loss
        pred['orig_z'] = ori_z
        return pred


class PoseVae(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.spatial = 1
        self.z_dim = 13 * args.bbox_dim
        self.args = args

        self.state_encoder = StateEncoder(512, self.z_dim)
        self.pose_decoder = PoseDecoder(self.z_dim * 2, self.z_dim)
        self.pose_encoder = PoseEncoder(self.z_dim * 2, self.z_dim)

    def forward(self, batch, test_sample=False):
        batch = vid_batch_to_cuda(batch)
        image = batch['image']  # (dt, V, 1, C, H, W)
        dt, V, _, C, H, W = image.size()
        bbox = batch['bbox'].view(dt, V, -1)    # (dt, V, O * 2)

        hidden_feat = self.state_encoder(image[0].view(V, C, H, W), bbox[0])  # (1, V, D)

        vel_bbox = bbox[1:] - bbox[0:-1]
        z, kl_loss, ori_z = self.pose_encoder((hidden_feat, bbox[1:], vel_bbox, test_sample))  # (dt - 1, V, De) -> (V, De)

        hidden_feat, v_pred = self.pose_decoder((hidden_feat, bbox[1:], z))  # (dt - 1, V, De)

        bbox_list = []
        cur_pose = bbox[0]
        for t in range(dt - 1):
            cur_pose = cur_pose + v_pred[t]
            bbox_list.append(cur_pose.clone())
        bbox = torch.stack(bbox_list, dim=0).view(dt - 1, V, 13, self.args.bbox_dim)

        pred = {}
        pred['v_pred'] = v_pred.view(dt - 1, V, 13, self.args.bbox_dim) # (dt - 1, V, 13 * 2)
        pred['bbox'] = bbox
        pred['kl_loss'] = kl_loss
        pred['orig_z'] = ori_z
        return pred


class PoseDecoder(nn.Module):
    def __init__(self, inp_nc, out_nc, nlayers=1):
        super().__init__()
        self.lstm = nn.LSTM(inp_nc, out_nc, nlayers)

    def forward(self, x):
        hid, box, latent = x
        T, V, D = box.size()
        latent = latent.unsqueeze(0)

        out_list = []
        pose = box[0:1]
        hid = (hid, hid)
        for t in range(T):
            inp = torch.cat([pose, latent], dim=-1)
            vel, hid = self.lstm(inp, hid)
            pose = vel + pose
            out_list.append(pose.clone())
        out = torch.cat(out_list, dim=0)  # (T, V, D)
        return hid, out


class PoseFCDecoder(nn.Module):
    def __init__(self, inp_nc, out_nc, nlayers=4, normalization='none', activation='relu'):
        super().__init__()
        dim_list = (inp_nc + out_nc,) * (nlayers) + (out_nc,)
        self.net = build_mlp(dim_list, batch_norm=normalization, activation=activation, final_nonlinearity=False)

    def forward(self, x):

        hid, box, latent = x
        T, V, D = box.size()
        latent = latent.unsqueeze(0).expand(T, V, D)
        inp = torch.cat([box, latent], dim=-1)
        out, hidden = self.lstm(inp, (hid, hid))

        return hidden, out


class PoseEncoder(nn.Module):
    def __init__(self, inp_nc, out_nc, nlayers=1):
        super().__init__()
        self.inp_nc = inp_nc
        self.out_nc = out_nc

        self.lstm = nn.LSTM(inp_nc, out_nc, nlayers)
        self.mu_head = nn.Linear(out_nc, out_nc)
        self.logvar_head = nn.Linear(out_nc, out_nc)

    def forward(self, x):
        hid, pos, vel, test_sample = x
        inp = torch.cat([pos, vel], dim=-1) # (dt, v, d)
        out, hidden = self.lstm(inp, (hid, hid)) # (dt, v, d)
        out = out[-1]  # (v, d)

        mu = self.mu_head(out)
        logvar = self.logvar_head(out)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        z = self.reparameterize(mu, logvar, test_sample)

        return z, kl_loss, mu

    def reparameterize(self, mu, logvar, force_sample):
        """generate sample from N(mu, var). or no sample"""
        if self.training or force_sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  ## (0, 1)
            return eps.mul(std).add_(mu)
            # return mu
        else:
            # mu is encoded by (src, dst) of the current data point.
            return mu


class StateEncoder(nn.Module):
    def __init__(self, feat_dim, pose_dim, nlayers=2, normalization='none', activation='relu'):
        super().__init__()
        self.pose_dim = pose_dim
        self.feat_dim = feat_dim

        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-1]
        self.enc = nn.Sequential(*modules)
        dim_list = (feat_dim + pose_dim, ) + (pose_dim, ) * (nlayers)
        self.linear = build_mlp(dim_list, batch_norm=normalization, activation=activation, final_nonlinearity=True)

    def forward(self, image, poses):
        """
        :param image: (V, C, H, W)
        :param poses: (V, Dpos)
        :return: (1, V, Do)
        """
        V = image.size(0)

        feat = self.enc(image).view(V, -1)
        feat_pose = torch.cat([feat, poses], dim=-1)
        out = self.linear(feat_pose)
        out = out.view(1, V, self.pose_dim)
        return out

