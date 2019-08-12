# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import imageio
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter
import pickle as pkl


class Evaluator(object):
    def __init__(self, model_name, args, name=None, metric=None):
        super().__init__()
        # save_dir = os.path.join(args.output_dir)
        # self.save_dir = os.path.join(save_dir, model_name)
        self.save_dir = model_name
        self.save_dir = os.path.join(self.save_dir, name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('## Make Directory: ', self.save_dir)

        self.image_size = args.image_size
        self.canvas_H = 256
        self.canvas_W = 256
        self.m_w = 5
        self.obj_size = args.obj_size
        self.metric = metric

        self.error_cnt = {}
        self.pix_cnt = {}
        cmd = 'rm -rf %s' % self.save_dir
        # print(cmd)
        os.system(cmd)
        self.tf_wr = SummaryWriter(self.save_dir)

    def perceptual_metric(self, im0, im1):
        """
        :param im0: (dt, V, 1, 3, H, W) in mean=(0, 0, 0) std=(1, 1, 1,)
        """
        dt, V, _, _, h, w = im0.size()
        assert V == 1
        im0 = im0.view(dt * V, 3, h, w)
        im1 = im1.view(dt * V, 3, h, w)
        d = self.metric.forward(im0, im1)  # dt * v
        d = d.reshape([dt])
        return d

    def save_seq2gif(self, image_list, index, pref='', suff=''):
        # save_file = os.path.join(self.save_dir, index) + '_%s.gif' % (suff)
        save_file = self.save_file_name(pref=pref, index=index, suff=suff, ext='.gif')
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
            print('## Make Dir', save_file)
        image_list = [cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB) for image in image_list]
        imageio.mimsave(save_file, image_list)

    def save_seqlist2gif(self, image_list_list, index, pref='', suff=''):
        # [[], []]
        save_file = self.save_file_name(pref=pref, index=index, suff=suff, ext='.gif')
        image_list = []
        for t in range(len(image_list_list[0])):
            merge = None
            for n in range(len(image_list_list)):
                merge = vstack((merge, image_list_list[n][t]))
            image_list.append(merge)

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
            print('## Make Dir', save_file)
        image_list = [cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB) for image in image_list]
        imageio.mimsave(save_file, image_list)

    def save_seqlist2traj(self, image_list_list, index, pref='', suff='', decay=0.8, canvas=None):
        # [[], []]
        save_file = self.save_file_name(pref=pref, index=index, suff=suff, ext='.jpg')
        merge = None
        for n in range(len(image_list_list)):
            this = None
            for t in range(len(image_list_list[0])):
                if this is None:
                    this = image_list_list[n][t]
                else:
                    this = this * decay + image_list_list[n][t]
            this += canvas * 0.35
            merge = vstack((merge, this))

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
            print('## Make Dir', save_file)
        cv2.imwrite(save_file, merge)

    def save_cmp_snapshot(self, all, index, stride=1, pref='', suff=''):
        merge = None
        start = divmod(len(all[0]) - 1, stride)[1]
        for i in range(start, len(all[0]), stride):
            one_step = all[0][i]  # vstack((all[0][i], all[1][i]))
            for j in range(1, len(all)):
                one_step = vstack((one_step, all[j][i]))
            merge = hstack((merge, one_step))
        save_file = self.save_file_name(pref=pref, index=index, suff=suff)
        cv2.imwrite(save_file, merge)

    def save_raw_box_image(self, obj, index, pref='', suff=''):
        save_file = self.save_file_name(pref=pref, index=index, suff=suff, ext='.pkl')
        with open(save_file, 'wb') as fp:
            pkl.dump(obj, fp)

    def save_file_name(self, pref, index, suff='', ext='.jpg'):
        index = index.split('/')[0]

        save_file = os.path.join(self.save_dir, pref + index + '_' + suff) + ext
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
            print('## Make Dir', save_file)
        return save_file

    def save_seq2traj(self, image_list, index, pref='', suff='', decay=0.9):
        image = image_list[0]
        for i in range(1, len(image_list)):
            image = decay * image + image_list[i]
        save_file = self.save_file_name(pref, index, suff)
        cv2.imwrite(save_file, image)

    def save_pack(self, image_list, traj_list, index, stride=1, pref=''):
        # self.save_cmp_snapshot(image_list, index, stride, pref=pref, suff='snap')
        self.save_seqlist2gif(image_list, index, pref=pref, suff='snap')
        self.save_vid_traj(traj_list, index, pref=pref, suff='box')

    def save_vid_traj(self, pred_list, index, suff='', pref=''):
        """
        :param pred_list: list of image: [im0, im2, ...]
        :return:
        """
        H = pred_list[0].shape[0]
        m = 5
        margin = np.ones([m, H, 3]).astype(np.uint8) * 255
        merge = pred_list[0]
        for i in range(1, len(pred_list)):
            merge = np.vstack((merge, margin, pred_list[i]))

        save_file = self.save_file_name(pref=pref, index=index, suff=suff)
        cv2.imwrite(save_file, merge)
        return

    def save_image(self, image, save_dir, name):
        save_file = os.path.join(save_dir, name) + '.jpg'
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
            print('## Make Dir', save_file)
        cv2.imwrite(save_file, image)

    def calc_total_bbox_error(self, gt_bbox, pred_bbox):
        dt, V, O, _ = gt_bbox.size()
        assert pred_bbox.size(0) == dt
        gt = gt_bbox[:, :, :, 0:2].cpu().detach().numpy()
        pred = pred_bbox[:, :, :, 0:2].cpu().detach().numpy()

        error = np.sum(np.square(gt - pred), axis=-1)  # (dt, V, O)
        error = np.mean(error, axis=1)  # (dt, O)
        return np.mean(error, axis=-1)  # (dt, )

    def cum_vid_bbox_error(self, gt_bbox, pred_bbox, save_steps=None):
        dt, V, O, _ = gt_bbox.size()
        assert pred_bbox.size(0) == dt
        if save_steps is None:
            save_steps = range(dt)
        gt = gt_bbox[:, :, :, 0:2].cpu().detach().numpy()
        pred = pred_bbox[:, :, :, 0:2].cpu().detach().numpy()

        error = np.sum(np.square(gt - pred), axis=-1)  # (dt, V, O)
        error = np.mean(error, axis=1)
        assert len(error) == dt
        for t in save_steps:
            self.update_error_cnt(error[t], t, 'box_center')

    def push_pix_error(self, err_list, method, suff):
        for t in range(len(err_list)):
            key = '%s_%s_%d' % (suff, method, t)
            if key not in self.pix_cnt:
                self.pix_cnt[key] = []
            self.pix_cnt[key].append(err_list[t])

    def push_box_error(self, err_list, method, suff):
        for t in range(len(err_list)):
            key = '%s_%s_%d' % (suff, method, t)
            if key not in self.error_cnt:
                self.error_cnt[key] = []
            self.error_cnt[key].append(err_list[t])

    def cum_perc_sim(self, im0, im1, suff):
        sim = self.perceptual_metric(im0, im1)
        dt = sim.shape[0]
        method = 'perc'

        all_key = '%s_%s_total' % (suff, method)
        if all_key not in self.pix_cnt:
            self.pix_cnt[all_key] = []

        for t in range(dt):
            for t in range(dt):
                err = sim[t]
                key = '%s_%s_%d' % (suff, method, t)
                if key not in self.pix_cnt:
                    self.pix_cnt[key] = []
                self.pix_cnt[key].append(err)
                self.pix_cnt[all_key].append(err)

    def update_error_cnt(self, error, dt, pref):
        """error: (O, ). Update total-dt, oi-dt... """
        for i in range(len(error)):
            key = pref + '_ball%d_%d' % (i, dt)
            if key not in self.error_cnt:
                self.error_cnt[key] = []
            self.error_cnt[key].append(error[i])

        key = pref + '_total_%d' % (dt)
        if key not in self.error_cnt:
            self.error_cnt[key] = []
        self.error_cnt[key].append(np.mean(error))

    def draw_save_error(self):
        save_file = os.path.join(self.save_dir, 'error_cnt.pkl')
        with open(save_file, 'wb') as fp:
            pkl.dump(self.error_cnt, fp)
            print('save to ', save_file)
        for key in self.error_cnt:
            name = '/'.join(key.split('_')[0:-1])
            t = int(key.split('_')[-1])
            value = 1. * np.mean(self.error_cnt[key])
            self.tf_wr.add_scalar(name, value, t)

    def draw_save_pix(self, stride=0):
        print('draw save pix')
        save_file = os.path.join(self.save_dir, 'pix_cnt.pkl')
        with open(save_file, 'wb') as fp:
            pkl.dump(self.pix_cnt, fp)
            print('save to ', save_file)
        for key in self.pix_cnt:
            if 'total' in key:
                continue
            name = '/'.join(key.split('_')[0: -1])
            t = int(key.split('_')[-1])
            value = np.mean(self.pix_cnt[key])
            self.tf_wr.add_scalar(name, value, t)

    def dump_error(self, iter, dset):
        for key in self.error_cnt:
            name = '/'.join(key.split('_'))
            name = dset + '/' + name
            value = np.mean(self.error_cnt[key])
            self.tf_wr.add_scalar(name, value, iter)

            self.error_cnt[key] = []

def vstack(img_list):
    m = 10
    width = -1
    if img_list[0] is None:
        return img_list[1]

    for img in img_list:
        if width < img.shape[1]:
            width = img.shape[1]
    new_img_list = []
    for img in img_list:
        zeros = np.ones([img.shape[0], width - img.shape[1], 3]) * 255
        new_img_list.append(np.hstack((img, zeros)))
    merge = new_img_list[0]
    for i in range(1, len(new_img_list)):
        img = new_img_list[i]
        zeros = np.ones([m, width, 3]) * 255
        merge = np.vstack((merge, zeros, img))
    return merge


def hstack(img_list):
    m = 10
    if img_list[0] is None:
        return img_list[1]
    h = img_list[0].shape[0]
    merge = img_list[0]
    for i in range(1, len(img_list)):
        img = img_list[i]
        zeros = np.ones([h, m, 3]) * 255
        merge = np.hstack((merge, zeros, img))
    return merge
