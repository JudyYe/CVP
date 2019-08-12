# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

"""
Utilities for logging in tensorboard.
"""


class Logger(object):
    def __init__(self, model_name, args):
        self.model_name = os.path.basename(model_name)
        self.plotter_dict = {}

        save_dir = os.path.join(args.output_dir)
        self.save_dir = os.path.join(save_dir, model_name, 'train')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('## Make Directory: ', self.save_dir)

        cmd = 'rm -rf %s' % self.save_dir
        os.system(cmd)
        self.tf_wr = SummaryWriter(self.save_dir)

    def set_tf_wr(self, tf_wr):
        self.tf_wr = tf_wr

    def add_loss(self, t, dictionary, pref='train/'):
        for key in dictionary:
            name = pref + key
            self.tf_wr.add_scalar(name, dictionary[key], t)

    def add_histogram(self, t, z, name=''):
        self.tf_wr.add_histogram(name, z, t)

    def add_hist_by_dim(self, t, z, name=''):
        dim = z.size(-1)
        for d in range(dim):
            index = name + '/%d' % d
            self.tf_wr.add_histogram(index, z[:, d], t)

    def add_images(self, iteration, images, name=''):
        images = torch.stack(images, dim=0)
        x = vutils.make_grid(images)
        self.tf_wr.add_image(name, x / 256, iteration)

    def print(self, t, epoch, losses, total_loss):
        print('[Epoch %2d] iter: %d of model' % (epoch, t), self.model_name)
        print('\tTotal Loss: %.6f' % total_loss)
        for k in losses:
            print('\t\t%s: %.6f' % (k, losses[k]))
