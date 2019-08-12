# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os
import glob

src_dir = '/scratch/yufeiy2/Penn_Action/frames/'
dst_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/penn_vis/'

np.random.seed(123)


def cp_K(K=100):
    img_list = glob.glob(os.path.join(src_dir, '*/*.jpg'))
    idx = np.random.permutation(len(img_list))

    for k in range(K):
        i = idx[k]
        fname = '_'.join(img_list[i].split('/')[-2:])
        dst_file = os.path.join(dst_dir, fname)
        if not os.path.exists(os.path.dirname(dst_file)):
            os.makedirs(os.path.dirname(dst_file))
        cmd = 'cp %s %s ' % (img_list[i], dst_file)
        os.system(cmd)


if __name__ == '__main__':
    cp_K()