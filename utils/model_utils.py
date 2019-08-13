# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import os

import torch


def vid_batch_to_cuda(batch):
    for k in batch:
        if k not in  ['index', 'info', 'bg_feat'] and batch[k] is not None:
            batch[k] = batch[k].cuda()
    return batch


def get_model_name(args):
    name = '%s/' % args.exp
    name += '%s%s_%s' % (args.dataset, args.modality, args.mod)
    name += '_%s' % (args.encoder)
    name += '_%s' % (args.decoder)
    # name += '%d' % args.dec_dims[-1]
    name += '_%s_%s' % (args.graph, args.gconv_unit_type)
    if args.appr_fea_loss:
        name += '_Pfea%g' % args.appr_loss_weight
    if args.appr_pix_loss:
        name += '_Ppix%g' % args.l1_dst_loss_weight
    return name


def build_loaders(args):
    if args.dataset.startswith('ss'):
        args.bbox_num = int(args.dataset[2])
        from data.ShapeStacks import build_vid_loaders
    elif args.dataset.startswith('penn'):
        args.bbox_num = 13
    elif args.dataset.startswith('pok'):
        args.bbox_num = 13
        from data.PennPoseKnows import build_vid_loaders
    elif args.dataset.startswith('demo'):
        from data.DemoImage import build_vid_loaders
    loader = build_vid_loaders(args)
    return loader



def build_all_model(args):
    if 'pok' in args.mod:
        if args.mod == 'pokVaePos':
            from .pok_model import PoseVaePos as model
            model = model(args)
        elif args.mod == 'pokFGan':
            from .pok_model import PoseFrameGan as model
            model = model(args)
    else:
        from cvp.models import ModelFactory
        model = ModelFactory[args.mod](args)
        if not args.is_train:
            print('Init...', args.checkpoint)
            pretrained_dict = torch.load(args.checkpoint)
            model.load_state_dict(pretrained_dict['model_state'])
            model.eval()
        model.cuda()

    return model
