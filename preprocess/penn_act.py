# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import os
import scipy.io as sio
import glob
import cv2
import pickle as pkl
    
label_dir = '/scratch/yufeiy2/Penn_Action/labels/'
split_dir = '/scratch/yufeiy2/Penn_Action/splits/'
rgb_dir = '/scratch/yufeiy2/Penn_Action/frames/'
save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/penn_vis/'
bone_dir = '/scratch/yufeiy2/Penn_Action/bones/'

kAll = 2326
def enum():
    action_list = {}
    for v in range(1, kAll+1):
        fname = label_dir + '%04d.mat' % v
        anno = sio.loadmat(fname)
        act = str(anno['action'][0])
        if act not in action_list:
            action_list[act] = 0
        action_list[act] += 1

    for each in action_list:
        print(each, action_list[each])


def vis(vid_name, X, Y, nframes, act, strip=4, idx=None):

    if idx is None:
        idx = range(0, nframes, strip)
    for f in idx:
        img = cv2.imread(rgb_dir + '%s/%06d.jpg' % (vid_name, f + 1))
        for j in range(X.shape[1]):
            x = int(X[f, j])
            y = int(Y[f, j])
            # if x + y < 3:
            #     print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
        draw_skeleton(img, X[f], Y[f]) # 10 * 4

        save_file = os.path.join(save_dir + '%s/%s/%06d.jpg' % (act, vid_name, f))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        cv2.imwrite(save_file, img)


def save_pose(strip, maxnum=16):
    subclass = 'tennis_serve'
    train = {}
    split = {1: [], -1: []}
    illegal = 0
    min_frame = -1
    for v in range(1, kAll+1):
        fname = label_dir + '%04d.mat' % v
        anno = sio.loadmat(fname)
        nframes = anno['nframes'][0][0]
        act = str(anno['action'][0])
        train[anno['train'][0][0]] = 0
        X = anno['x'].astype(np.float)
        Y = anno['y'].astype(np.float)
        if act != subclass:
            continue
        if min_frame == -1 or min_frame > nframes:
            min_frame = nframes
        if len(np.where(X + Y <= 3)[0]) > 0:
            # idx = np.where(X + Y <=3)
            # print(X[idx], Y[idx])
            # vis('%04d' % v, X, Y, nframes, act, strip=1, idx=np.where(X + Y <= 3)[0])
            # illegal += 1
            continue

        split[anno['train'][0][0]].append(v)
        bone_vid = []
        for f in range(0, nframes, strip):
            bone = get_skeleton(X[f], Y[f])
            bone_vid.append(bone)

        bone_vid = np.array(bone_vid, dtype=np.float)
        h, w, _ = anno['dimensions'][0]
        bone_vid[:, :, 0] = bone_vid[:, :, 0] / w
        bone_vid[:, :, 2] = bone_vid[:, :, 2] / w
        bone_vid[:, :, 1] = bone_vid[:, :, 1] / h
        bone_vid[:, :, 3] = bone_vid[:, :, 3] / h

        save_file = os.path.join(bone_dir, '%04d.npz' % v)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
            print('## Make Direcotry: ', save_file)
        np.savez_compressed(save_file, bone=bone_vid)
        # print(bone_vid.shape)
    print('illegal', illegal)
    save_file = split_dir + '%s.npz' % subclass
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'wb') as fp:
        pkl.dump(split, fp)
    print(len(split[-1]), len(split[1]))
    print('min frame', min_frame)

# 1.  head
# 2.  left_shoulder  3.  right_shoulder
# 4.  left_elbow     5.  right_elbow
# 6.  left_wrist     7.  right_wrist
# 8.  left_hip       9.  right_hip
# 10. left_knee      11. right_knee
# 12. left_ankle     13. right_ankle
#

def draw_skeleton(img, x, y):
    x = x.astype(np.int)
    y = y.astype(np.int)

    bone = []
    m = 15
    # head
    # l, r = min(x[0], x[1], x[2]) - m, max(x[0], x[1], x[2]) + m
    # t, b = y[0] - m, min(y[1], y[2]) - m
    l, r = x[0] -  m, x[0] +  m
    t, b = y[0] -  m, y[0] +  m
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])

    # body
    l, r = min(x[1], x[2], x[7], x[8]) - m, max(x[1], x[2], x[7], x[8]) + m
    t, b = min(y[1], y[2], y[7], y[8]) - m, max(y[1], y[2], y[7], y[8]) + m
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])

    # left upper arm
    l, r, t, b = bbox_from_2pt(x[1], x[3], y[1], y[3], m)
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])
    # right upper arm
    l, r, t, b = bbox_from_2pt(x[2], x[4], y[2], y[4], m)
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])

    # left lower arm
    l, r, t, b = bbox_from_2pt(x[5], x[3], y[5], y[3], m)
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])
    # right lower arm
    l, r, t, b = bbox_from_2pt(x[6], x[4], y[6], y[4], m)
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])

    # left upper leg
    l, r, t, b = bbox_from_2pt(x[7], x[9], y[7], y[9], m)
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])
    # right upper leg
    l, r, t, b = bbox_from_2pt(x[8], x[10], y[8], y[10], m)
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])

    # left lower leg
    l, r, t, b = bbox_from_2pt(x[9], x[11], y[9], y[11], m)
    b += m
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])
    # right lower leg
    l, r, t, b = bbox_from_2pt(x[10], x[12], y[10], y[12], m)
    b += m
    cv2.rectangle(img, (l, t), (r, b), (0,255,0), 1)
    bone.append([l, t, r, b])

    return bone


def get_skeleton(x, y):
    x = x.astype(np.int)
    y = y.astype(np.int)

    bone = []
    m = 15
    # head
    # l, r = min(x[0], x[1], x[2]) - m, max(x[0], x[1], x[2]) + m
    # t, b = y[0] - m, min(y[1], y[2]) - m
    l, r = x[0] -  m, x[0] +  m
    t, b = y[0] -  m, y[0] +  m
    bone.append([l, t, r, b])

    l, r = min(x[1], x[2], x[7], x[8]) - m, max(x[1], x[2], x[7], x[8]) + m
    t, b = min(y[1], y[2], y[7], y[8]) - m, max(y[1], y[2], y[7], y[8]) + m
    bone.append([l, t, r, b])

    # left upper arm
    l, r, t, b = bbox_from_2pt(x[1], x[3], y[1], y[3], m)
    bone.append([l, t, r, b])
    # right upper arm
    l, r, t, b = bbox_from_2pt(x[2], x[4], y[2], y[4], m)
    bone.append([l, t, r, b])

    # left lower arm
    l, r, t, b = bbox_from_2pt(x[5], x[3], y[5], y[3], m)
    bone.append([l, t, r, b])
    # right lower arm
    l, r, t, b = bbox_from_2pt(x[6], x[4], y[6], y[4], m)
    bone.append([l, t, r, b])

    # left upper leg
    l, r, t, b = bbox_from_2pt(x[7], x[9], y[7], y[9], m)
    bone.append([l, t, r, b])
    # right upper leg
    l, r, t, b = bbox_from_2pt(x[8], x[10], y[8], y[10], m)
    bone.append([l, t, r, b])

    # left lower leg
    l, r, t, b = bbox_from_2pt(x[9], x[11], y[9], y[11], m)
    b += m
    bone.append([l, t, r, b])
    # right lower leg
    l, r, t, b = bbox_from_2pt(x[10], x[12], y[10], y[12], m)
    b += m
    bone.append([l, t, r, b])

    return bone


def bbox_from_2pt(x1, x2, y1, y2, m):
    l, r = min(x1, x2) - m, max(x1, x2) + m
    t, b = min(y1, y2) - m, max(y1, y2) + m
    return l, r, t, b

if __name__ == '__main__':
    # enum()
    save_pose(strip=1)