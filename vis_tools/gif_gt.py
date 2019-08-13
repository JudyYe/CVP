# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import imageio
import numpy as np
import os
import cv2
import argparse
import glob

save_dir = '/scratch/yufeiy2/shapestacks/ss_gt_vis/'
gt_dir = '/scratch/yufeiy2/shapestacks/frc_35/'


def make_gif(index):
    search = os.path.join(gt_dir, index, '*.jpg')
    gif_list = glob.glob(search)
    gif_list = sorted(gif_list)

    dst_file = os.path.join(save_dir, index + '_init.jpg')
    if not os.path.exists(os.path.dirname(dst_file)):
        os.makedirs(os.path.dirname(dst_file))
    cmd = 'cp %s %s' % (gif_list[0], dst_file)
    os.system(cmd)

    gif_list = np.array(gif_list)[1:16]
    img_list = [cv2.imread(fname) for fname in gif_list]
    img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
    dst_file = os.path.join(save_dir, index + '_snap.gif')
    imageio.mimsave(dst_file, img_list)

def gif_all():
    with open(flist) as fp:
        index_list = [line.strip() for line in fp]
    for i, index in enumerate(index_list):
        make_gif(index)
        if i % 50 == 0:
            print('%d / %d' % (i, len(index_list)))


if __name__ == '__main__':
    # num_list = [4, 5, 6]
    num_list = [3]
    for kNum in num_list:
        # flist = '/scratch/yufeiy2/shapestacks/splits/env_ccs+blocks-hard+easy-h=%d-vcom=1+2+3+4+5+6-vpsf=0/eval.txt' % kNum
        flist = '/scratch/yufeiy2/shapestacks/splits/env_ccs+blocks-hard+easy-h=%d-vcom=1+2+3-vpsf=0/eval.txt' % kNum

        gif_all()

