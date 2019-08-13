# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .graph import GraphFactory
from .vid_encoder import EncoderFactory
from .decoder import DecoderFactory
from utils.model_utils import vid_batch_to_cuda
from .layers import build_fblock


class BaseBG(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.dt = args.dt

        self.dec_dst_gt = args.dec_dst_gt
        self.decoder_target_size = args.obj_size
        self.bbox_dim = args.bbox_dim
        self.stop_grad = args.gconv_stop_grad
        self.gconv_unit_type = args.gconv_unit_type

        self.fc7_dim = args.embedding_dim
        self.appr_dim = args.appr_dim
        self.feat_constraint = args.feat_constraint

        # build image tower
        resent = models.resnet18(pretrained=True)
        modules = nn.ModuleList(list(resent.children())[:-2])
        self.image_tower = modules
        self.skip_layer = args.dec_skip
        self.embed_dim_list = []
        for skip in self.skip_layer:
            self.embed_dim_list.append(modules[skip][1].bn2.num_features)

        # build head
        appr_kwargs = {
            'dim_list': [args.embedding_dim, ] + args.fac_hidden_dims + [args.appr_dim],
            'batch_norm': args.normalization,
            'activation': args.activation,
            'final_pool': self.final_pool
        }
        self.appr_net = build_fblock(**appr_kwargs)

        # build encoder
        enc_kwargs = {
            'feat_dim_list': [args.embedding_dim, ],
            'output_dim': args.noise_dim,
            'hidden_dims': args.enc_hidden_dims,
            'norm': 'none',
            'act': args.activation,
            'dt': self.dt,
            'long_term': args.enc_long_term,
        }
        self.encoder = EncoderFactory[args.encoder](**enc_kwargs)
        # build decoder
        dec_kwargs = {
            'feat_dim_list': [args.appr_dim, self.bbox_dim * self.num_bbox, self.embed_dim_list],
            'dims': args.dec_dims,
            'normalization': args.dec_norm,
            'activation': args.dec_act,
            'radius': args.radius / args.image_size[0],
        }
        self.decoder = DecoderFactory[args.decoder.split('_')[0]](**dec_kwargs)

        # build gcn
        gconv_kwargs = {
            'feat_dim_list': [args.appr_dim, self.bbox_dim * self.num_bbox],
            'noise_dim': args.noise_dim,
            'num_blocks': args.gconv_num_blocks,
            'pooling': args.gconv_pooling,
            'preact_normalization': args.gconv_normalization,
            'unit_type': args.gconv_unit_type,
            'spatial': self.spatial,
            'stop_grad': self.stop_grad,
        }
        self.graph_net = GraphFactory[args.graph](**gconv_kwargs)

    def forward(self, in_vecs):
        vid_batch = vid_batch_to_cuda(in_vecs)
        # 1. Feature Extraction: 'feats' = 'appr' | 'bbox'
        vid_batch = self._forward_image_encode(vid_batch)
        # 2. get z, kl_loss, using only I_0, I_dt-1
        img_z, kl_loss, long_u = self.encoder(vid_batch)

        preds = self._forward_with_z(vid_batch, img_z, self.dt)

        preds['kl_loss'] = kl_loss
        preds['orig_z'] = long_u
        return preds

    def _forward_with_z(self, vid_batch, img_z, time_len):
        preds = self._forward_n_step(vid_batch, img_z, time_len)
        bg_feat = self.get_bg_feat(vid_batch['bg_feat'], 0)
        src_recon = self.decoder(vid_batch['appr'], vid_batch['bbox'],
                                 self.decoder_target_size, bg_feat)
        dst_recon = self.decoder(preds['appr'], preds['bbox'],
                                 self.decoder_target_size, bg_feat)
        for key in src_recon:
            new_key = 'real_' + key
            preds[new_key] = src_recon[key]
        for key in dst_recon:
            new_key = 'pred_' + key
            preds[new_key] = dst_recon[key]
        return preds

    def forward_with_reality(self, intput, time_len):
        """encode z / u with ground truth I_{t+T} w/o resample"""
        vid_batch = vid_batch_to_cuda(intput)
        vid_batch = self._forward_image_encode(vid_batch)

        # 2. get u, kl_loss, using f1, f2
        img_z, kl_loss, ori_z = self.encoder.no_sample(vid_batch)
        predictions = self._forward_with_z(vid_batch, img_z, time_len)
        return predictions

    def forward_inception(self, intput, time_len, seed=None):
        """sample z / u from a distribution"""
        vid_batch = vid_batch_to_cuda(intput)
        vid_batch = self._forward_image_encode(vid_batch)

        # 2. get u, kl_loss, using f1, f2
        V = vid_batch['bbox'].size(1)
        img_z = self.encoder.batch_sample(V, time_len, seed, vid_batch['image'][0])
        predictions = self._forward_with_z(vid_batch, img_z, time_len)
        return predictions

    def _forward_n_step(self, vid_batch, img_z, time_len):
        cur_frame = self.filter_time_stamp(0, vid_batch)
        trip = self._next_trip(cur_frame['bbox'], cur_frame['trip'])
        _, V, D = img_z.size()
        predictions = []
        for t in range(time_len):
            # 3.1 get z^{t, t+1} to predict frame. (V, D) -> (V, O, D)
            obj_z = img_z[t].unsqueeze(1)
            stop = True if (self.stop_grad and t == 0) else False
            feat_pred = self.graph_net(cur_frame['appr'], cur_frame['bbox'], obj_z, trip, stop)
            feat_pred['appr'] = self._apply_feat_constraint(feat_pred['appr'])
            # 'feats', 'appr', 'bbox'
            cur_frame.update(feat_pred)
            # trip = vid_batch['trip'][t]  # (V, T, D)
            predictions.append(feat_pred.copy())
        predictions = collate_batch(predictions, time_len, V)  # 'feats', 'appr', 'bbox',
        return predictions

    def _apply_feat_constraint(self, feat):
        feat = F.normalize(feat, dim=-3)
        return feat

    def _forward_image_encode(self, input):
        raise NotImplementedError

    def filter_time_stamp(self, t, vid_batch):
        filtered = {}
        for key in vid_batch:
            if vid_batch[key] is None:
                filtered[key] = None
            else:
                filtered[key] = vid_batch[key][t]
        return filtered

    def get_bg_feat(self, bg: list, t):
        for i in range(len(bg)):
            bg[i] = bg[i][t]
        return bg

    def _unsqz(self, tensor, dim=0):
        return tensor.unsqueeze(dim)


class CVP(BaseBG):
    def __init__(self, args):
        self.num_bbox = 1
        self.spatial = 2
        self.final_pool = 4
        super().__init__(args)

    def _forward_image_encode(self, vid):
        dt, V, o, C, H, W = vid['crop'].size()
        feats = self._forward_tower(vid['crop'].view(-1, C, H, W))  # (dt * V * o, 512, W, W)

        appr = self.appr_net(feats) #

        appr = appr.view(dt, V, o, self.appr_dim, self.spatial, self.spatial)
        appr = self._apply_feat_constraint(appr)

        vid['appr'] = appr

        bg_feat = self._forward_tower(vid['image'].view(-1, C, H, W), True)  # (dt * V * 1, 512, 7, 7)
        for i in range(len(bg_feat)):
            _, d, h, w = bg_feat[i].size()
            bg_feat[i] = bg_feat[i].view(dt, V, 1, d, h, w)
        vid['bg_feat'] = bg_feat

        return vid

    def _forward_tower(self, image, save_skip=False):
        """image: N, C, H, W"""
        skip = []
        for i, layer in enumerate(self.image_tower):
            image = layer(image)
            if i in self.skip_layer:
                skip.append(image)
        if save_skip:
            skip.reverse()
            return skip
        else:
            return image

    def _next_trip(self, bbox, cur_trip):
        """(V, O, 2), (V, T, 3)"""
        return cur_trip


class LP(CVP):
    def __init__(self, args):
        super().__init__(args)

    def forward_with_reality(self, intput, time_len):
        return self.forward_lp(intput, time_len, False)

    def forward_inception(self, intput, time_len, seed=None):
        return self.forward_lp(intput, time_len, True)

    def forward_lp(self, intput, time_len, sample):
        """todo: clean!"""
        vid_batch = vid_batch_to_cuda(intput)
        # 1. 'feats' = 'appr' | 'bbox', with image_tower AND factorize
        vid_batch = self._forward_image_encode(vid_batch)
        V = vid_batch['image'].size(1)
        bg_feat = self.get_bg_feat(vid_batch['bg_feat'], 0)

        predictions = []
        cur_frame = self.filter_time_stamp(0, vid_batch)
        cur_frame['pred_recon'] = vid_batch['image'][1] # reality
        trip = self._next_trip(cur_frame['bbox'], cur_frame['trip'])
        self.encoder.init_hidden(vid_batch)
        for t in range(time_len):
            img_z, kl_loss, orig_z = self.encoder.one_sample(cur_frame['pred_recon'], sample)
            # img_z: dt=1, V, D -> (V, O, D)
            obj_z = img_z[0].unsqueeze(1)
            stop = True if (self.stop_grad and t == 0) else False
            feat_pred = self.graph_net(cur_frame['appr'], cur_frame['bbox'], obj_z, trip, stop)
            feat_pred['appr'] = self._apply_feat_constraint(feat_pred['appr'])

            out = self.decoder(self._unsqz(cur_frame['appr'], 0),
                               self._unsqz(cur_frame['bbox'], 0), self.decoder_target_size, bg_feat)
            feat_pred['pred_recon'] = out['recon'].squeeze(0)

            cur_frame.update(feat_pred)
            predictions.append(feat_pred.copy())

        predictions = collate_batch(predictions, time_len, V)

        src_recon = self.decoder(vid_batch['appr'], vid_batch['bbox'],
                                 self.decoder_target_size, bg_feat)
        for key in src_recon:
            new_key = 'real_' + key
            predictions[new_key] = src_recon[key]
        return predictions


class NoFactor(BaseBG):
    def __init__(self, args):
        self.num_bbox = args.bbox_num
        self.spatial = 7
        self.final_pool = None
        super().__init__(args)

    def _forward_image_encode(self, vid):
        dt, V, o, C, H, W = vid['image'].size()
        feats = self._forward_tower(vid['image'].view(-1, C, H, W))  # (dt * V * o, 512, W, W)

        appr = self.appr_net(feats) #

        appr = appr.view(dt, V, o, self.appr_dim, self.spatial, self.spatial)
        appr = self._apply_feat_constraint(appr)

        vid['appr'] = appr

        bg_feat = self._forward_tower(vid['image'].view(-1, C, H, W), True)  # (dt * V * 1, 512, 7, 7)
        for i in range(len(bg_feat)):
            _, d, h, w = bg_feat[i].size()
            bg_feat[i] = bg_feat[i].view(dt, V, 1, d, h, w)
        vid['bg_feat'] = bg_feat
        return vid

    def _forward_tower(self, image, save_skip=False):
        """image: N, C, H, W"""
        skip = []
        for i, layer in enumerate(self.image_tower):
            image = layer(image)
            if i in self.skip_layer:
                skip.append(image)
        if save_skip:
            skip.reverse()
            return skip
        else:
            return image

    def _forward_n_step(self, vid_batch, img_z, time_len):
        cur_frame = self.filter_time_stamp(0, vid_batch)
        cur_frame['bbox'] = self._scatter_box(cur_frame['bbox'], cur_frame['valid'])
        # trip = vid_batch['trip'][0]  # (V, T, D)
        trip = self._next_trip(cur_frame['bbox'], cur_frame['trip'])
        _, V, D = img_z.size()
        predictions = []
        for t in range(time_len):
            # 3.1 get z^{t, t+1} to predict frame. (V, D) -> (V, O, D)
            obj_z = img_z[t].unsqueeze(1)
            stop = True if (self.stop_grad and t == 0) else False
            feat_pred = self.graph_net(cur_frame['appr'], cur_frame['bbox'], obj_z, trip, stop)
            feat_pred['appr'] = self._apply_feat_constraint(feat_pred['appr'])
            # 'feats', 'appr', 'bbox'
            cur_frame.update(feat_pred)
            predictions.append(feat_pred.copy())
            predictions[-1]['bbox'] = self._gather_box(cur_frame['bbox'], cur_frame['valid'])
        predictions = collate_batch(predictions, time_len, V)  # 'feats', 'appr', 'bbox',
        return predictions

    def _scatter_box(self, bbox, valid):
        """
        :param bbox: (V, O, D)
        :param valid: (V, 1, O)
        :return: (V, num, D)
        """
        V, O, D = bbox.size()
        holder = torch.zeros(V, self.num_bbox, D).to(bbox)
        valid = valid.view(V, O, 1).expand(V, O, D)
        holder = holder.scatter_add(1, valid, bbox)
        return holder

    def _gather_box(self, holder, valid):
        """
        :param holder: (V, num, D)
        :param valid: (V, 1, O)
        :return: (V, O, D)
        """
        V, _, D = holder.size()
        O = valid.size(-1)
        valid = valid.view(V, O, 1).expand(V, O, D) # (V, O, D)
        bbox = holder.gather(1, valid)
        return bbox

    def _next_trip(self, bbox, cur_trip):
        return None


def collate_batch(batch_list, dt, V):
    batch = {}
    for b, pred in enumerate(batch_list):
        for key in pred:
            if not key in batch:
                batch[key] = []
            batch[key].append(pred[key])
    for key in batch:
        if batch[key][0] is None:
            continue
        else:
            batch[key] = torch.stack(batch[key], 0)
    return batch


ModelFactory = {
    'cvp': CVP,
    'lp': LP,
    'noFactor': NoFactor,
}