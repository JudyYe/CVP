# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossManager(object):
    def __init__(self, args):
        self.summerize_total_losses = None
        self.separate_losses = None
        self.args = args
        self.D = self.args.embedding_dim

        if args.mod in ['cvp', 'noFactor']:
            self.separate_losses = self._separate_losses
        elif args.mod in ['pokVaePos']:
            self.summerize_total_losses = self._pose_vae_loss
        elif args.mod in ['pokFGan']:
            self.separate_losses = self._pose_gan_loss
        else:
            print('loss not defined in mod: ', args.mod)
            raise NotImplementedError

    def _separate_losses(self, truth, predictions):
        """losses are comprised of 1. KL loss 2. prediction loss 3. reconstruction loss"""
        dt = truth['bbox'].size(0)
        pred_loss = torch.zeros(1).to(predictions['kl_loss'])
        ae_loss = torch.zeros(1).to(predictions['kl_loss'])
        losses = {}
        # 1. encoder: kl
        ae_loss = add_loss(ae_loss, predictions['kl_loss'], losses, 'kl_loss', self.args.kl_loss_weight)

        # 2. feat: appr | bbox
        recon_loss = resize_l1_loss(predictions['pred_recon'][0:dt - 1], truth['image'][1: dt])
        pred_loss = add_loss(pred_loss, recon_loss, losses, 'appr_pixel_loss', self.args.l1_dst_loss_weight)
        # bbox:
        bbox_loss = F.mse_loss(predictions['bbox'][0:dt - 1], truth['bbox'][1:dt])
        pred_loss = add_loss(pred_loss, bbox_loss, losses, 'bbox_loss', self.args.bbox_loss_weight)

        # 3. decoder: l1 recon loss
        recon_loss = resize_l1_loss(predictions['real_recon'], truth['image'])
        ae_loss = add_loss(ae_loss, recon_loss, losses, 'recon_loss', self.args.l1_src_loss_weight)
        return [pred_loss, ae_loss], losses

    def _pose_gan_loss(self, truth, predictions):
        g_loss = torch.zeros(1).to(truth['image'])
        d_loss = torch.zeros(1).to(truth['image'])
        losses = {}
        V = truth['image'].size(1)

        real_label = torch.zeros([V, 2]).to(truth['bbox'])
        fake_label = torch.zeros([V, 2]).to(truth['bbox'])
        real_label[:, 0] = real_label[:, 0] + 1
        fake_label[:, 1] = fake_label[:, 1] + 1

        # G
        loss = nn.BCELoss()(predictions['fake_label'], real_label)
        g_loss = add_loss(g_loss, loss, losses, 'G:fake_label', 1)
        loss = F.l1_loss(predictions['fake'], truth['image'])
        g_loss = add_loss(g_loss, loss, losses, 'G:l1', 100)

        # D
        loss = nn.BCELoss()(predictions['real_label'], real_label)
        d_loss = add_loss(d_loss, loss, losses, 'D:real', 1)
        loss = nn.BCELoss()(predictions['fake_label'], fake_label)
        d_loss = add_loss(d_loss, loss, losses, 'D:fake', 1)
        return [d_loss, g_loss], losses

    def _pose_vae_loss(self, truth, predictions):
        ae_loss = torch.zeros(1).to(predictions['kl_loss'])
        losses = {}
        # 1. encoder: kl
        ae_loss = add_loss(ae_loss, predictions['kl_loss'], losses, 'kl_loss', self.args.kl_loss_weight)
        # 2. pose vel loss
        pos_loss = F.mse_loss(predictions['bbox'], truth['bbox'][1:])
        ae_loss = add_loss(ae_loss, pos_loss, losses, 'vel_loss', self.args.pose_loss_weight)

        return ae_loss, losses, None


def resize_l1_loss(pred, gt, mask=None):
    """Resize gt to the size of pred. and calculate loss"""
    dt, V, O, C, H, W = pred.size()
    gt_W = gt.size(-1)
    factor = gt_W // W
    l1_target = F.avg_pool2d(gt.view(dt * V * O, C, gt_W, gt_W), kernel_size=factor, stride=factor)
    if mask is not None:
        l1_target = l1_target.view(dt, V, O, C, H, W) * mask
        print(mask[0, 0, :, 0, 0, 0], l1_target[0, 0, :, 0, 0, 0], pred[0, 0, :, 0, 0, 0])
        pass
    loss = F.l1_loss(pred, l1_target.view(dt, V, O, C, H, W))
    return loss


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    """register loss to total_loss and loss_dict"""
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def get_G_loss(truth, predictions, losses, args):
    g_loss = torch.zeros(1).to(truth['image'])
    V = truth['image'].size(1)

    real_label = torch.zeros([V, 2]).to(truth['bbox'])
    fake_label = torch.zeros([V, 2]).to(truth['bbox'])
    real_label[:, 0] = real_label[:, 0] + 1
    fake_label[:, 1] = fake_label[:, 1] + 1
    # G
    if 'fake_label' in predictions:
        loss = nn.BCELoss()(predictions['fake_label'], real_label)
        g_loss = add_loss(g_loss, loss, losses, 'G:fake_label', 1)
    loss = F.l1_loss(predictions['fake'], truth['image'])
    g_loss = add_loss(g_loss, loss, losses, 'G:l1', args.l1_dst_loss_weight)

    return g_loss, losses


def get_D_loss(truth, predictions, losses, args):
    d_loss = torch.zeros(1).to(truth['image'])
    V = truth['image'].size(1)

    real_label = torch.zeros([V, 2]).to(truth['bbox'])
    fake_label = torch.zeros([V, 2]).to(truth['bbox'])
    real_label[:, 0] = real_label[:, 0] + 1
    fake_label[:, 1] = fake_label[:, 1] + 1

    loss = nn.BCELoss()(predictions['real_label'], real_label)
    d_loss = add_loss(d_loss, loss, losses, 'D:real', 1)
    loss = nn.BCELoss()(predictions['fake_label'], fake_label)
    d_loss = add_loss(d_loss, loss, losses, 'D:fake', 1)
    return d_loss, losses
