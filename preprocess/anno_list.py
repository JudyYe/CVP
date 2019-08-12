# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os

image_dir = '/scratch/yufeiy2/bair/images'
list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/bair_push/'
save_dir = '/scratch/yufeiy2/bair_anno/'

with open(list_dir + 'bair_trainimglist.txt') as fp:
    image_list = [line.strip() for line in fp]

np.random.seed(123)
idx = np.random.permutation(len(image_list))
image_list = np.array(image_list)
image_list = image_list[idx[0:300]]

for i, index in enumerate(image_list):
    src_path = os.path.join(image_dir, index) + '.jpg'
    dst_path = os.path.join(save_dir, index) + '.jpg'

    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))
    cmd = 'cp %s %s' % (src_path, dst_path)
    os.system(cmd)


with open(list_dir + 'bair_annoimglist.txt', 'w') as fp:
    for img in image_list:
        fp.write('%s\n' % img)
