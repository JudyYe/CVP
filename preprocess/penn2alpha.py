# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os
import glob
import pickle as pkl


# python3 demo.py --list examples/list-coco-demo.txt --indir ${img_directory} --outdir examples/res --save_img

src_dir = '/scratch/yufeiy2/Penn_Action/frames/'
# list_dir = '/scratch/yufeiy2/Penn_Action/splits/'
list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/penn_action'

# src_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo/imgs/'
# list_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo/'
# output_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo/joints/'

def make_list():
    img_list = glob.glob(os.path.join(src_dir, '*/*.jpg'))
    img_list = sorted(img_list)
    list_file = os.path.join(list_dir, 'img_list.txt')
    wr_fp = open(list_file, 'w')
    print(list_file)
    for v, each in enumerate(img_list):
        vid = each.split('/')[-2]
        name = each.split('/')[-1]

        wr_fp.write('%s/%s\n' % (vid, name))
        if v > 200:
            break
    wr_fp.close()

# with open(list_dir + 'all.pkl', 'rb') as fp:
#     obj = pkl.load(fp)
# print(obj)
# {'drum': {1: [], -1: []}}


if __name__ == '__main__':
    make_list()
    # make_one_list()