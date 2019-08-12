# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob

import numpy as np
import os

# list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/'
# data_dir = '/scratch/yufeiy2/sthsth/v1/'

list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/robot_push/'
data_dir = '/scratch/yufeiy2/robot_push/'


def count_box(all_file):
    with open(list_dir + all_file) as fp:
        img_list = [line.strip() for line in fp]
    save_file = list_dir + all_file.split('.')[0] + '_num.txt'
    wr_fp = open(save_file, 'w')
    num_dict = {}
    for i, index in enumerate(img_list):
        feat_path = os.path.join(data_dir, 'feats', index + '.npz')
        if not os.path.exists(feat_path):
            num = -1
        else:
            try:
                obj = np.load(feat_path)
            except AttributeError:
                print('????', feat_path)
                continue
            bboxes = obj['bboxes']
            num = len(bboxes)
        wr_fp.write('%s %d\n' % (index, num))
        num_dict[index] = num
        if i % 50000 == 0:
            print('%d / %d' % (i, len(img_list)))
    wr_fp.close()n
    return num_dict


def batch_list(num_dict, obj_file):
    print('batch listing...')
    vid_img = {}
    with open(list_dir + obj_file) as fp:
        for line in fp:
            index = line.strip()
            vid = os.path.dirname(index)
            if not vid_img.has_key(vid):
                vid_img[vid] = []
            vid_img[vid].append(index)

    lower_bound_list = [1]
    upper_bound_list = [3, 5, -1]
    strip = 3
    dt_list = [1, 3, 6, 12]

    for l_th in lower_bound_list:
        for u_th in upper_bound_list:
            for dt in dt_list:
                new_base = 'robo_%s_l%du%ddt%d.txt' % (obj_file.split('_')[1], l_th, u_th, dt)
                save_file = list_dir + new_base
                wr_fp = open(save_file, 'w')
                cnt = 0
                for vid in vid_img:
                    img_list = sorted(vid_img[vid])
                    for s_idx in range(0, len(img_list) - dt, strip):
                        e_idx = s_idx + dt
                        if l_th >= 0 and (num_dict[img_list[s_idx]] < l_th or num_dict[img_list[e_idx]] < l_th):
                            continue
                        if u_th > 0 and (num_dict[img_list[s_idx]] > u_th or num_dict[img_list[e_idx]] > u_th):
                            continue
                        wr_fp.write('%s %s\n' % (img_list[s_idx], img_list[e_idx]))
                        cnt += 1
                wr_fp.close()
                print('Done ', save_file, 'line: %d' % cnt)


def ignore_subclass():
    name_list = glob.glob(list_dir + 'robo_subclass_*.txt')
    for name in name_list:
        basename = os.path.basename(name)
        last_two = basename.split('_')[-2:]
        basename = 'sth_' + '_'.join(last_two)
        new_file = os.path.dirname(name) + '/' + basename

        cmd = 'mv %s %s' % (name, new_file)
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    # ignore_subclass()
    # all_file = 'sth_pushing_imglist.txt'
    all_file = 'robo_imglist.txt'
    num_dict = count_box(all_file)
    batch_list(num_dict, all_file)
