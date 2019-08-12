# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models

from .layers import build_mlp



class VidEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, long_term, hidden_dims=None, norm='none', act='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.long_term = long_term

        if hidden_dims is None:
            layers = [input_dim * 2, output_dim]
            self.encoder1 = None
            self.mu_head = nn.Linear(layers[0], output_dim)
            self.logvar_head = nn.Linear(layers[0], output_dim)
        else:
            layers = (input_dim * 2,) + hidden_dims
            self.encoder1 = build_mlp(layers, batch_norm=norm, activation=act, final_nonlinearity=True)
            self.mu_head = nn.Linear(layers[-1], output_dim)
            self.logvar_head = nn.Linear(layers[-1], output_dim)

    def forward(self, vid_batch):
        """
        Called during training. 1. encode to u 2. resample
        :return: obj_z: (Dt, V, D), kl_loss: criterion
        """
        obj_z, kl_loss, ori_z = self._forward(vid_batch, True)
        return obj_z, kl_loss, ori_z

    def no_sample(self, vid_batch):
        """
        Called during realistic testing. 1. encode u. 2. NO sample
        :return: obj_z: (Dt, V, D), kl_loss: criterion
        """
        obj_z, kl_loss, ori_z = self._forward(vid_batch, False)
        return obj_z, kl_loss, ori_z

    def _forward(self, vid_batch, sample=True):
        raise NotImplementedError

    def batch_sample(self, V, time_len, seed, image):
        """
        Called during inception. Don't encode. sample from N(0, 1)
        :param V:
        :param time_len:
        :param seed:
        :return:
        """
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.randn(V, self.output_dim).unsqueeze(0).cuda()  # (1, V, D)
        img_z = self.generate_z_from_u(u, time_len, V)
        return img_z

    def reparameterize(self, mu, logvar, sample):
        """generate sample from N(mu, var). or no sample"""
        if self.training and sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  ## (0, 1)
            return eps.mul(std).add_(mu)
            # return mu
        else:
            # mu is encoded by (src, dst) of the current data point.
            return mu

    def mu_logvar(self, input_feat):
        out = input_feat
        if self.encoder1 is not None:
            out = self.encoder1(input_feat)
        mu = self.mu_head(out)
        logvar = self.logvar_head(out)

        # mu = nn.functional.normalize(mu)
        return mu, logvar

    def generate_z_from_u(self, u, dt, V):
        raise NotImplementedError


class TrajHierarchy(VidEncoder):
    def __init__(self, feat_dim_list, output_dim, dt, long_term, hidden_dims, norm, act):
        """under construction!!!"""
        assert len(feat_dim_list) == 1
        input_dim = feat_dim_list[0]
        super().__init__(input_dim, output_dim, long_term, hidden_dims, norm, act)
        self.dt = dt

        self.lstm_layers = 1
        self.lstm = nn.LSTM(output_dim, output_dim, self.lstm_layers)

        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-1]
        self.image_tower = nn.Sequential(*modules)

    def _forward(self, vid, sample=True):
        dt, V, _, C, H, W = vid['image'].size()
        src_image = vid['image'][0].squeeze(1)
        dst_image = vid['image'][self.long_term].squeeze(1)
        src_feat = self.image_tower(src_image).view(V, -1)
        dst_feat = self.image_tower(dst_image).view(V, -1)  ## (1, 512)
        src_dst = torch.cat([src_feat, dst_feat], dim=-1)
        long_u_mu, logvar = self.mu_logvar(src_dst)
        reparam = self.reparameterize(long_u_mu, logvar, sample).unsqueeze(0)  # (1, V, D)

        # (Dt, V, D)
        img_z = self.generate_z_from_u(reparam, dt, V)
        kl_loss = -0.5 * torch.mean(1 + logvar - long_u_mu.pow(2) - logvar.exp())
        return img_z, kl_loss, long_u_mu

    def generate_z_from_u(self, u, dt, V):
        """
        :param u: (1, V, D)
        :return: (dt, V, D)
        """
        h0 = torch.zeros(u.size()).to(u)
        zeros = torch.zeros(u.size()).to(u).expand(dt, V, self.output_dim)
        img_z, (h0, c0) = self.lstm(zeros, (h0, u))
        return img_z


class ImageNoZ(VidEncoder):
    def __init__(self, feat_dim_list, output_dim, dt, long_term, hidden_dims, norm, act):
        """under construction!!!"""
        assert len(feat_dim_list) == 1
        input_dim = feat_dim_list[0]
        super().__init__(input_dim, output_dim, long_term, hidden_dims, norm, act)
        self.dt = dt

        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-1]
        self.image_tower = nn.Sequential(*modules)

    def _forward(self, vid, sample=True):
        dt, V, _, C, H, W = vid['image'].size()

        src_image = vid['image'][0].squeeze(1)
        dst_image = vid['image'][self.long_term].squeeze(1)

        src_feat = self.image_tower(src_image).view(V, -1)
        dst_feat = self.image_tower(dst_image).view(V, -1)
        src_dst = torch.cat([src_feat, dst_feat], dim=-1)

        long_u_mu, logvar = self.mu_logvar(src_dst)
        reparam = self.reparameterize(long_u_mu, logvar, sample).unsqueeze(0)  # (1, V, D)

        # (Dt, V, D)
        img_z = self.generate_z_from_u(reparam, self.dt, V)
        kl_loss = -0.5 * torch.mean(1 + logvar - long_u_mu.pow(2) - logvar.exp())
        return img_z, kl_loss, long_u_mu

    def generate_z_from_u(self, u, dt, V):
        """
        :param u: (1, V, D)
        :return: (dt, V, D)
        """
        img_z = u.expand(dt, V, self.output_dim)
        return img_z


class ImageFixPrior(VidEncoder):
    def __init__(self, feat_dim_list, output_dim, dt, long_term, hidden_dims=None, norm='batch',
                 act='relu'):
        """under construction!!!"""
        assert len(feat_dim_list) == 1
        input_dim = feat_dim_list[0]
        super().__init__(input_dim, output_dim, long_term, hidden_dims, norm, act)
        self.dt = dt

        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-1]
        self.image_tower = nn.Sequential(*modules)

        self.hidden = None

    def _forward(self, vid, sample=True):
        dt, V, _, C, H, W = vid['image'].size()

        image = vid['image'].view(dt * V, C, H, W)
        feat = self.image_tower(image).view(dt, V, self.input_dim)
        src_feat = feat[0: dt - 1]
        dst_feat = feat[1: dt]
        src_dst = torch.cat([src_feat, dst_feat], dim=-1)

        long_u_mu, logvar = self.mu_logvar(src_dst.view(-1, 2 * self.input_dim))  # (N, noise)
        reparam = self.reparameterize(long_u_mu.view(dt - 1, V, self.output_dim), logvar.view(dt - 1, V, self.output_dim),
                                      sample).view(dt, V, self.output_dim)  # (dt - 1, V, D)
        # (Dt, V, D)
        kl_loss = -0.5 * torch.mean(1 + logvar - long_u_mu.pow(2) - logvar.exp())
        return reparam, kl_loss, long_u_mu

    def batch_sample(self, V, time_len, seed, image):
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.randn(time_len, V, self.output_dim).cuda()  # (dt, V, D)
        return u

    def reparameterize(self, mu, logvar, sample):
        """
        :param u: (dt - 1, V, D)
        :return: (dt, V, D). random generalize one
        """
        dt, V, D = mu.size()
        dt += 1
        if self.training and sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  ## (0, 1)
            orig = eps.mul(std).add_(mu)
            last = torch.randn(1, V, D).to(mu)
            img_z = torch.cat([orig, last], dim=0)
            return img_z
        else:
            # mu is encoded by (src, dst) of the current data point.
            last = torch.zeros([1, V, D]).to(mu)
            img_z = torch.cat([mu, last], dim=0)
            return img_z


class ImageLearnedPrior(nn.Module):
    """Stole from https://github.com/edenton/svg/blob/master/models/lstm.py """

    def __init__(self, feat_dim_list, output_dim, dt, **kwargs):
        """under construction!!!"""
        assert len(feat_dim_list) == 1
        input_dim = feat_dim_list[0]
        super().__init__()
        self.dt = dt

        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-1]
        self.image_tower = nn.Sequential(*modules)

        self.lstm_layers = 1
        self.prior_lstm = gaussian_lstm(input_dim, output_dim, output_dim, 1)
        self.posterior_lstm = gaussian_lstm(input_dim, output_dim, output_dim, 1)

    def forward(self, vid_batch):
        dt, V, o, C, H, W = vid_batch['image'].size()
        feats = self.image_tower(vid_batch['image'].view(dt * V, C, H, W)).view(dt, V, -1)
        h_target = feats
        h = feats[0: dt - 1]
        z_t, mu, logvar = self.posterior_lstm(h_target, dt)
        _, mu_p, logvar_p = self.prior_lstm(h, dt)

        kl_loss = kl_criterion(mu, logvar, mu_p, logvar_p)
        return z_t, kl_loss, mu

    def init_hidden(self, dummy=None):
        self.prior_lstm.init_hidden()
        self.posterior_lstm.init_hidden()

    def no_sample(self, vid_batch):
        """
        level == 0: POSTERIOR-GT
        level == 1: PRIOR-GT
        level == 3: PRED-PRIOR
        Called during realistic testing. 1. encode u. 2. NO sample
        :return: obj_z: (Dt, V, D), kl_loss: criterion
        """
        dt, V, o, C, H, W = vid_batch['image'].size()
        feats = self.image_tower(vid_batch['image'].view(dt * V, C, H, W)).view(dt, V, -1)
        z_t, mu_p, logvar_p = self.posterior_lstm(feats, dt, False)
        kl_loss = torch.mean(mu_p)

        return z_t, kl_loss, mu_p

    def one_sample(self, image, sample):
        V, o, C, H, W = image.size()
        feats = self.image_tower(image.view(V, C, H, W)).view(1, V, -1)
        z_t, mu_p, logvar_p = self.prior_lstm.forward_one(feats, sample)
        kl_loss = torch.mean(mu_p)

        return z_t, kl_loss, mu_p


class gaussian_lstm(nn.Module):
    """Stole from Emily https://github.com/edenton/svg"""
    def __init__(self, input_size, output_size, hidden_size, n_layers, reparam=True):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        V = 1
        h_u = (torch.zeros(1, V, self.hidden_size).cuda(), torch.zeros(1, V, self.hidden_size).cuda())
        return h_u

    def forward(self, input, true_dt, sample=True):
        dt, V, _ = input.size()
        embedded = self.embed(input.view(-1, self.input_size))
        embedded = embedded.view(dt, V, self.hidden_size)

        zeros = (torch.zeros(1, V, self.hidden_size).cuda(), torch.zeros(1, V, self.hidden_size).cuda())
        h_in, _ = self.lstm(embedded, zeros)  # (dt , V, D)
        if dt == true_dt:  # then it is posterior
            h_in = h_in[1:]
        mu = self.mu_net(h_in)  # (true_dt - 1, V, Dout)
        logvar = self.logvar_net(h_in)  # (dt - 1, V, Dout)
        z = self.reparameterize(mu, logvar, sample)  # (dt, V, Dout)
        return z, mu, logvar

    def forward_one(self, input, sample=True):
        dt, V, _ = input.size()
        embedded = self.embed(input.view(-1, self.input_size)).view(dt, V, self.hidden_size)
        h_in, self.hidden = self.lstm(embedded, self.hidden)  # (dt , V, D)
        mu = self.mu_net(h_in)  # (true_dt - 1, V, Dout)
        logvar = self.logvar_net(h_in)  # (dt - 1, V, Dout)
        z = self.reparameterize_one(mu, logvar, sample)  # (dt, V, Dout)
        return z, mu, logvar

    def reparameterize_one(self, mu, logvar, sample):
        """
        :param u: (1, V, D)
        :return: (1, V, D). random generalize one
        """
        dt, V, D = mu.size()
        assert dt == 1
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  ## (0, 1)
            img_z = eps.mul(std).add_(mu)
            return img_z
        else:
            return mu

    def reparameterize(self, mu, logvar, sample):
        """
        :param u: (dt - 1, V, D)
        :return: (dt, V, D). random generalize one
        """
        dt, V, D = mu.size()
        dt += 1
        if self.training or sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  ## (0, 1)
            orig = eps.mul(std).add_(mu)
            last = torch.randn(1, V, D).to(mu)
            img_z = torch.cat([orig, last], dim=0)
            return img_z
        else:
            # mu is encoded by (src, dst) of the current data point.
            last = torch.zeros([1, V, D]).to(mu)
            img_z = torch.cat([mu, last], dim=0)

            return img_z


def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
    return kld.mean()



EncoderFactory = {
    'traj': TrajHierarchy,
    'noZ': ImageNoZ,
    'fp': ImageFixPrior,
    'lp': ImageLearnedPrior,
}
