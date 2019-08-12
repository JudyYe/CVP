# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os
import cv2


save_dir = '/scratch/yufeiy2/syn_blocks/'
np.random.seed(123)
# save_dir = './tmp/'
def syn(width=224, num=5000):
    for n in range(num):
        color = np.random.uniform(size=[1, 1, 3])
        color *= 255
        img = np.zeros([width, width, 3]) + color
        fname = os.path.join(save_dir, '%04d.jpg' % n)
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
            print('## Make Dir', fname)
        cv2.imwrite(fname, img)

if __name__ == '__main__':
    syn()