# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob
import pandas as pd
import numpy as np
import os


# image_dir = '/scratch/yufeiy2/Moments_in_Time_Mini/frames/training/'
# vid_file = '/glusterfs/yufeiy2/pred_vid_data/moment_mini_vidlist.txt'
# save_file = '/glusterfs/yufeiy2/pred_vid_data/moment_mini.txt'

# image_dir = '/scratch/yufeiy2/thumos/UCF101_frame/'
# vid_dir = '/scratch/yufeiy2/thumos/UCF101/'
# vid_file = '/glusterfs/yufeiy2/pred_vid_data/ucf101_mini_vidlist.txt'
# save_file = '/glusterfs/yufeiy2/pred_vid_data/ucf101_mini.txt'

list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/'
# vid_dir = image_dir = '/scratch/yufeiy2/sthsth/v1/images/'
# vid_file = list_dir + 'sth_mini_vidlist.txt'
# save_file = list_dir + 'sth_mini.txt'

list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/robot_push/'
vid_dir = image_dir  = '/scratch/yufeiy2/robot_push/images/push_train/'
vid_file = list_dir + 'robo_mini_vidlist.txt'
img_file = list_dir + 'robo_mini_imglist.txt'

np.random.seed(123)
top_vid = 20


def make_ucf_vid():
    wr_fp = open(vid_file, 'w')
    vid_list = os.listdir(vid_dir)
    top_vid = 20
    mini_list = []
    idx = np.random.permutation(len(vid_list))[0: top_vid]
    for i in range(top_vid):
        v = idx[i]
        mini_list.append(vid_list[v])
        wr_fp.write('%s\n' % vid_list[v])
    wr_fp.close()

    wr_fp = open(img_file, 'w')
    for vid in mini_list:
        img_list = glob.glob(os.path.join(image_dir, vid, '*.jpg'))
        img_list = sorted(img_list)
        for img_path in img_list:
            index = os.path.join(vid, os.path.basename(img_path).split('.')[0])
            wr_fp.write('%s\n' % index)
            print(index)
    wr_fp.close()


def make_trim_image_list():
    with open(vid_file) as fp:
        vid_list = [line.strip().split('.')[0] for line in fp]
    wr_fp = open(save_file, 'w')
    for vid in vid_list:
        img_list = glob.glob(os.path.join(image_dir, vid, '*.jpg'))
        img_list = sorted(img_list)
        for img_path in img_list:
            index = os.path.join(vid, os.path.basename(img_path))
            wr_fp.write('%s\n' % index)
            print(index)
    wr_fp.close()

def make_sth_class_list():
    # action_list = ['putting']  #, 'pushing']
    # action_list = ['throwing']
    action_list = ['pushing']
    # action_list = ['opening', 'closing']
    # action_list = ['turning', 'opening', 'closing', 'throwing', 'putting', 'pulling']

    label_file = list_dir + 'something-something-v1-train.csv'
    vid_list = []
    with open(label_file) as fp:
        for line in fp:
            index, dscp = line.strip().split(';')
            dscp = str(dscp).lower()
            for a in action_list:
                if a in dscp:
                    vid_list.append((index, dscp))
                    break
    print('len: ', len(vid_list))
    vid_file = list_dir + 'sth_%s_vidlist.txt' % '_'.join(action_list)
    with open(vid_file, 'w') as fp:
        for each in vid_list:
            index, dscp = each
            fp.write('%s %s\n' % (index, dscp))
    image_file = list_dir + 'sth_%s_imglist.txt' % '_'.join(action_list)
    cnt = 0
    with open(image_file, 'w') as fp:
        for v, each in enumerate(vid_list):
            vid = each[0]
            img_list = glob.glob(image_dir + vid + '/*.jpg')
            # print(image_dir + vid + '*.jpg', len(img_list))
            img_list = sorted(img_list)
            for img_path in img_list:
                index = os.path.join(vid, os.path.basename(img_path).split('.')[0])
                fp.write('%s\n' % index)
                cnt += 1
    print('img list :', cnt)


def make_sth_pair_list(action_list, dt=12, strip=4):
    vid_file = list_dir + 'sth_subclass_%s_vidlist.txt' % '_'.join(action_list)
    pair_file = list_dir + 'sth_subclass_%s_pairlist.txt' % '_'.join(action_list)
    img_file = list_dir + 'sth_subclass_%s_pairimg.txt' % '_'.join(action_list)
    with open(vid_file) as fp:
        vid_list = [line.split()[0] for line in fp]
    wr_fp = open(img_file, 'w')
    with open(pair_file, 'w') as fp:
        cnt = 0
        for v, vid in enumerate(vid_list):
            img_list = glob.glob(image_dir + vid + '/*.jpg')
            img_list = sorted(img_list)
            for i in range(0, len(img_list) - dt, strip):
                s = os.path.join(vid, os.path.basename(img_list[i]).split('.')[0])
                d = os.path.join(vid, os.path.basename(img_list[i + dt]).split('.')[0])
                fp.write('%s %s\n' % (s, d))
                wr_fp.write('%s\n%s\n' % (s, d))
                cnt += 1
    print('pair: ', cnt)


def mini_list(old_file):
    top = 100
    new_file = list_dir + old_file.split('.')[0] + '_mini%d.txt' % top


    with open(list_dir + old_list) as fp:
        lines = [line.strip() for line in fp]
    np.random.seed(123)
    idx = np.random.permutation(len(lines))

    with open(new_file, 'w') as fp:
        for i in range(top):
            fp.write('%s\n' % lines[idx[i]])
    print('save to ', new_file)

def make_robot_list():
    vid_list = os.listdir(vid_dir)
    with open(vid_file, 'w') as fp:
        for vid in vid_list:
            fp.write('%s\n' % vid)
        print(vid)
    wr_fp = open(img_file, 'w')
    for vid in vid_list:
        img_list = glob.glob(vid_dir + vid + '/*.jpg')
        print(vid_dir + vid + '/*.jpg')
        img_list = sorted(img_list)
        for img_path in img_list:
            index = os.path.join(vid, os.path.basename(img_path).split('.')[0])
            wr_fp.write('%s\n' % index)
    wr_fp.close()


if __name__ == '__main__':
    # make_small_list()
    # make_ucf_vid()
    # make_trim_image_list()

    # make_sth_class_list()
    # old_list = 'sth_putting_l1u3dt3.txt'
    # mini_list(old_list)

    # action_list = ['putting']  #, 'pushing']
    # action_list = ['throwing']
    # make_sth_pair_list(action_list)

    make_ucf_vid()
    # make_robot_list()