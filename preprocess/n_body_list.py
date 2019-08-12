# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob

import numpy as np
import os

# list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/nbody_v2/'
# vid_pref = list_dir + '3body_'
# data_dir = '/scratch/yufeiy2/nbody_v2/'
# num_val = 100
# num_test = 100

# list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/nbody_v3/'
# vid_pref = list_dir + '3body_'
# data_dir = '/scratch/yufeiy2/nbody_v3/'
# num_val = 1000
# num_test = 1000

list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/nbody_v4/'
vid_pref = list_dir + '3body_'
data_dir = '/scratch/yufeiy2/nbody_v4/'
num_val = 1000
num_test = 1000

# list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/nbody_v5/'
# vid_pref = list_dir + '3body_'
# data_dir = '/scratch/yufeiy2/nbody_v5/'
# num_val = 1000
# num_test = 1000


def make_vid_list():
    vid_list = glob.glob(data_dir + '3_*')
    idx = np.random.permutation(len(vid_list))
    vid_list = np.array(vid_list)[idx]

    if not os.path.exists(os.path.dirname(vid_pref)):
        os.makedirs(os.path.dirname(vid_pref))
    val_fp = open(vid_pref + 'valvid.txt', 'w')
    test_fp = open(vid_pref + 'testvid.txt', 'w')
    train_fp = open(vid_pref + 'trainvid.txt', 'w')
    all_fp = open(vid_pref + 'allvid.txt', 'w')

    for v, vid in enumerate(vid_list):
        vid = os.path.basename(vid)
        if v < num_val:
            val_fp.write('%s\n' % vid)
        elif num_val <= v < num_test + num_val:
            test_fp.write('%s\n' % vid)
        else:
            train_fp.write('%s\n' % vid)
        all_fp.write('%s\n' % vid)


def make_vid_pair(vid_file):
    pair_file = vid_pref + vid_file.replace('vid', 'pair')
    wr_fp = open(pair_file, 'w')
    with open(vid_pref + vid_file) as fp:
        vid_list = [line.strip() for line in fp]

    for vid in vid_list:
        img_list = glob.glob(os.path.join(data_dir, vid, '*.jpg'))
        img_list = [os.path.join(vid, os.path.basename(each)).split('.')[0] for each in img_list]
        img_list = sorted(img_list)
        for i in range(1, len(img_list)):
            wr_fp.write('%s %s\n' % (img_list[i - 1], img_list[i]))
    wr_fp.close()


def make_longterm_pair(vid_file, dt):
    pair_file = vid_pref + vid_file.replace('vid', 'dt%02d_pair' % dt)
    wr_fp = open(pair_file, 'w')
    with open(vid_pref + vid_file) as fp:
        vid_list = [line.strip() for line in fp]

    for vid in vid_list:
        img_list = glob.glob(os.path.join(data_dir, vid, '*.jpg'))
        img_list = [os.path.join(vid, os.path.basename(each)).split('.')[0] for each in img_list]
        img_list = sorted(img_list)
        if dt > 0:
            for i in range(dt, len(img_list), dt):
                wr_fp.write('%s %s\n' % (img_list[i - dt], img_list[i]))
                if 'train' not in vid_file:
                    break
        else:
            wr_fp.write('%s %s\n' % (img_list[0], img_list[-1]))
    wr_fp.close()


def make_gt_list(vid_file, n=3):
    pair_file = vid_pref + vid_file.replace('vid', 'gt')
    wr_fp = open(pair_file, 'w')
    with open(vid_pref + vid_file) as fp:
        vid_list = [line.strip() for line in fp]

    for vid in vid_list:
        img_list = glob.glob(os.path.join(data_dir, vid, '*.jpg'))
        img_list = sorted(img_list)
        gt_np = np.load(os.path.join(data_dir, vid, 'meta.npz'))['meta']
        assert len(gt_np) == len(img_list)
        for i in range(len(img_list)):
            index = os.path.join(vid, os.path.basename(img_list[i]).split('.')[0])
            wr_fp.write('%s' % index)
            for g in range(n):
                wr_fp.write(' %.1f %.1f' % (gt_np[i, g, 0], gt_np[i, g, 1]))
            wr_fp.write('\n')

    wr_fp.close()

def make_unary_pair(vid_file, min_len):
    pair_file = vid_pref + vid_file.replace('vid', 'unary')
    wr_fp = open(pair_file, 'w')
    with open(vid_pref + vid_file) as fp:
        vid_list = [line.strip() for line in fp]

    for vid in vid_list:
        img_list = glob.glob(os.path.join(data_dir, vid, '*.jpg'))
        img_list = [os.path.join(vid, os.path.basename(each)).split('.')[0] for each in img_list]
        if len(img_list) < min_len:
            continue
        img_list = sorted(img_list)
        flag = True
        for i in range(min_len):
            if i != int(os.path.basename(os.path.basename(img_list[i])).split('.')[0]):
                print(i, img_list[i])
                flag = False
                break
        if not flag:
            break
        wr_fp.write('%s\n' % img_list[0])
    wr_fp.close()


if __name__ == '__main__':
    make_vid_list()

    vid_file_list = ['valvid.txt',
                     'testvid.txt',
                     'trainvid.txt']
    make_gt_list('allvid.txt')
    # dt_list = [1, 2, 4, 8, 16, -1]
    # for vid_file in vid_file_list:
    #     make_vid_pair(vid_file)
    #
    #     for dt in dt_list:
    #         make_longterm_pair(vid_file, dt)
    for vid_file in vid_file_list:
        make_unary_pair(vid_file, 16)