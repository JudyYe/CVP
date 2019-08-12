# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os
from .base_cfgs import BaseOptions, bool_flag


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # training iters
        self.parser.add_argument('--num_iterations', default=1e+5, type=int)
        self.parser.add_argument('--epoches', default=630, type=int)
        self.parser.add_argument('--learning_rate', default=1e-4, type=float)

        # losses
        self.parser.add_argument('--appr_pix_loss', default=True, type=bool_flag)
        self.parser.add_argument('--l1_dst_loss_weight', default=1, type=float)
        self.parser.add_argument('--appr_fea_loss', default=False, type=bool_flag)
        self.parser.add_argument('--appr_loss_weight', default=100, type=float)
        self.parser.add_argument('--l1_src_loss_weight', default=1, type=float)
        self.parser.add_argument('--kl_loss_weight', default=1e-2, type=float)
        self.parser.add_argument('--bbox_loss_weight', default=100, type=float)
        self.parser.add_argument('--adv_loss_weight', default=1e-3, type=float)

        self.parser.add_argument('--print_every', default=50, type=int)
        self.parser.add_argument('--curve_log_every', default=200, type=int)
        self.parser.add_argument('--image_log_every', default=1000, type=int)
        self.parser.add_argument('--evaluate_every', default=5000, type=int)

        self.isTrain = True
