# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os

demo_path = 'data/demo/demo_ss.txt'
with open(demo_path) as fp:
    index_list = [line.strip() for line in fp]

src_dir = '/scratch/yufeiy2/shapestacks/'
dst_dir = 'data/demo/'

for index in index_list:
    src_file = os.path.join(src_dir, 'frc_35', index, 'rgb-cam_1-%02d.jpg' % 0)
    dst_file = os.path.join(dst_dir, index + '.jpg')
    cmd = 'cp %s %s' % (src_file, dst_file)
    os.system(cmd)

    src_file = os.path.join(src_dir, 'frc_35', index, 'cam_1.npy')
    bbox = np.load(src_file)[0]
    dst_file = os.path.join(dst_dir, index + '.npy')
    np.save(dst_file, bbox)
