# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os

from .base_cfgs import BaseOptions, bool_flag


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # For testing
        self.parser.add_argument('--test_mod', default='multi', help='multi/best_100')
        self.parser.add_argument('--test_split', default='test', type=str)
        self.parser.add_argument('--checkpoint', default=None, type=str)
        self.parser.add_argument('--num_val_samples', default=None, type=int)
        self.parser.add_argument('--shuffle_val', default=False, type=bool_flag)
        self.isTrain = False


