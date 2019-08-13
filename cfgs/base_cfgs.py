# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import ast
import os

import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.isTrain = None

    def initialize(self):
        MY_OUTPUT = './output/'

        parser = argparse.ArgumentParser()
        parser.add_argument('--mod', default='cvp', choices=['cvp', 'noFactor', 'lp',
                                                                 'pokVaeVel', 'pokVaePos', 'pokFGan',])
        # Optimization hyperparameters
        parser.add_argument('--bs', default=32, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--is_train', default=True)

        # for Dataset
        parser.add_argument('--dataset', default='ss3')
        parser.add_argument('--modality', default='rgb')
        parser.add_argument('--image_size', default='224, 224', type=int_tuple)
        # parser.add_argument('--obj_size', default='32, 32', type=int_tuple)
        parser.add_argument('--obj_size', default='224, 224', type=int_tuple)
        parser.add_argument('--loader_num_workers', default=4, type=int)
        parser.add_argument('--dt', default=16, type=int)
        parser.add_argument('--strip', default=4, type=int)
        parser.add_argument('--radius', default=35, type=int)

        parser.add_argument('--normalization', default='none')
        parser.add_argument('--activation', default='relu')

        # Encoder options
        parser.add_argument('--encoder', default='traj', type=str)
        parser.add_argument('--enc_reparam', default=True, type=bool)
        parser.add_argument('--enc_hidden_dims', default='64', type=int_tuple)
        parser.add_argument('--enc_long_term', default=-1, type=int)
        parser.add_argument('--appr_dim', default=32, type=int)
        parser.add_argument('--noise_dim', default=8, type=int)

        # GCN options
        parser.add_argument('--graph', default='fact_gc', type=str)
        parser.add_argument('--embedding_dim', default=512, type=int)
        parser.add_argument('--gconv_unit_type', default='n2e2n', type=str)
        # parser.add_argument('--gconv_dim', default=512, type=int)
        parser.add_argument('--gconv_num_units', default=2, type=int)
        parser.add_argument('--gconv_num_blocks', default=4, type=int)
        parser.add_argument('--gconv_pooling', default='avg', type=str)
        parser.add_argument('--gconv_normalization', default='none', type=str)
        parser.add_argument('--gconv_activation', default='relu', type=str)
        parser.add_argument('--gconv_stop_grad', default=True, type=bool_flag)
        parser.add_argument('--recon_stop_grad', default=True, type=bool_flag)
        parser.add_argument('--dec_zero_grad', default=False, type=bool_flag)

        # Factor options
        parser.add_argument('--cnn_last_relu', default='none', type=str)
        parser.add_argument('--cnn_norm_feat', default=True, type=bool_flag)
        parser.add_argument('--feat_constraint', default='norm', type=str)
        parser.add_argument('--bbox_dim', default=2, type=int)
        parser.add_argument('--bbox_num', default=3, type=int)
        parser.add_argument('--fac_hidden_dims', default='[512]', type=int_list)

        # Decoder options
        parser.add_argument('--decoder', default='comb_late', type=str)
        parser.add_argument('--dec_dims', default='256,128,64,32,16,8', type=int_tuple)  # * 32
        parser.add_argument('--dec_norm', default='batch')
        parser.add_argument('--dec_act', default='leakyrelu')
        parser.add_argument('--dec_skip', default='7,6,5', type=int_tuple)
        parser.add_argument('--dec_dst_gt', default=True, type=bool_flag)

        # Output options
        parser.add_argument('--output_dir', default=MY_OUTPUT)
        parser.add_argument('--exp', default='tmp')

        self.parser = parser
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu
        self.opt = self.set_default_args(self.opt)
        return self.opt

    def set_default_args(self, args):
        if self.isTrain:
            args.batch_size = args.bs // args.dt
            if args.decoder.startswith('comb'):
                dec_dims = {'early': (256,128,64,32),
                            'mid': (256,128,64,32,16),
                            'late': (256,128,64,32,16,8),
                            'comb': args.dec_dims}
                args.dec_dims = dec_dims[args.decoder.split('_')[-1]]
        else:
            model_path = args.checkpoint
            pretrained_dict = torch.load(model_path)
            args_in_checkpoint = pretrained_dict['args']
            from .collections import AttrDict
            args = AttrDict(args.__dict__)
            skip_key = ['dataset', 'enc_long_term']
            for key in args_in_checkpoint:
                if key not in args:
                    continue
                if key in skip_key:
                    continue
                args[key] = args_in_checkpoint[key]
            args.batch_size = 1
        args.is_train = self.isTrain
        return args


def int_tuple(s):
    if s == 'none':
        x = None
    else:
        x = tuple(int(i) for i in s.split(','))
    return x


def int_list(s):
    return ast.literal_eval(s)


def float_tuple(s):
    return tuple(float(i) for i in s.split(','))


def str_tuple(s):
    return tuple(s.split(','))


def bool_flag(s):
    if s == '1' or s == 'True':
        return True
    elif s == '0' or s == 'False':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)
