# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob

import numpy as np
import os
import cv2
import imageio

def mv_all():
    src_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/20bn-something-something-v2-{:02d}'
    for i in range(5, 20):
        dst_dir = '/scratch/yufeiy2/robot_push/push/'
        cmd = 'mv %s %s &' % (src_dir.format(i), dst_dir)
        os.system(cmd)


def copy_all():
    res_dir = '/scratch/yufeiy2/pred_vid/'
    dst_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/vis_sth/'
    dir_list = glob.glob(os.path.join(res_dir, '*'))
    for each in dir_list:
        src = os.path.join(each, 'vis')
        index = os.path.basename(each)

        dst = os.path.join(dst_dir, index)
        cmd = 'cp -r %s %s & ' % (src, dst)
        print(cmd)
        os.system(cmd)

def gif_all(max_num, strip=4):
    np.random.seed(123)
    rgb_dir = '/scratch/yufeiy2/Penn_Action/frames/'
    save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/gif_penn/'
    kAll = 2326
    for v in range(kAll):
        vid_dir = rgb_dir + '%04d/' % (v)
        frame_list = glob.glob(vid_dir + '*.jpg')
        if len(frame_list) <= max_num * strip + 1:
            continue
        start = np.random.randint(0, len(frame_list) - max_num * strip)
        rgb_list = []
        for i in range(max_num):
            idx = start + 1 + strip * i
            fname = vid_dir + '%06d.jpg' % idx
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_list.append(img)

        save_file = os.path.join(save_dir, '%04d.gif' % v)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        imageio.mimsave(save_file, rgb_list)

if __name__ == '__main__':
    # copy_all()
    gif_all(8)

