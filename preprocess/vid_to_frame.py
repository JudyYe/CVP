# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os
import cv2

# vid_file = '/glusterfs/yufeiy2/pred_vid_data/moment_mini_vidlist.txt'
# vid_dir = '/scratch/yufeiy2/Moments_in_Time_Mini/training/'
# img_dir = '/scratch/yufeiy2/Moments_in_Time_Mini/frames/training/'

vid_file = '/glusterfs/yufeiy2/pred_vid_data/ucf101_mini_vidlist.txt'
vid_dir = '/scratch/yufeiy2/thumos/UCF101/'
img_dir = '/scratch/yufeiy2/thumos/UCF101_frame/'

def extract_frame(vid):
    frame_dir = os.path.join(img_dir, vid.split('.')[0])
    vidcap = cv2.VideoCapture(os.path.join(vid_dir, vid))
    success, image = vidcap.read()
    count = 0
    while success:
        name = frame_dir + '/' + '%05d.jpg' % count
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
            print('## Make Directory: ', name)
        cv2.imwrite(name, image)  # save frame as JPEG file
        success, image = vidcap.read()
        if count % 100 == 0:
            print('Read a new frame: ', success)
        count += 1


def extract_video_list():
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]
    for vid in vid_list:
        extract_frame(vid)


if __name__ == '__main__':
    extract_video_list()