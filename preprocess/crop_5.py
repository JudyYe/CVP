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
crop_dir = '/scratch/yufeiy2/Penn_Action/joint/'
kAll = 2326


def enum():
    action_list = {}
    for v in range(1, kAll + 1):
        fname = label_dir + '%04d.mat' % v
        anno = sio.loadmat(fname)
        act = str(anno['action'][0])
        if act not in action_list:
            action_list[act] = 0
        action_list[act] += 1

    for each in action_list:
        print(each, action_list[each])


def vis(vid_name, bones, nframes, act, strip=4, idx=None):
    if idx is None:
        idx = range(0, nframes, strip)

    for f in idx:
        img = cv2.imread(crop_dir + '%s/%06d.jpg' % (vid_name, f + 1))
        h, w = img.shape[0], img.shape[1]
        for j in range(bones.shape[1]):
            x = int((bones[f, j, 0] + bones[f, j, 2]) / 2 * w)
            y = int((bones[f, j, 1] + bones[f, j, 3]) / 2 * h)
            # if x + y < 3:
            #     print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
        draw_skeleton(img, bones[f], w, h)  # 13 * 2

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
    for v in range(1, kAll + 1):
        fname = label_dir + '%04d.mat' % v
        anno = sio.loadmat(fname)
        nframes = anno['nframes'][0][0]
        act = str(anno['action'][0])
        train[anno['train'][0][0]] = 0
        if act != subclass:
            continue
        if min_frame == -1 or min_frame > nframes:
            min_frame = nframes
        joint_file = os.path.join(bone_dir, '%04d_cropJoint.npz' % v)
        if not os.path.exists(joint_file):
            continue
        joint = np.load(joint_file)['bone']
        X = joint[:, :, 0]
        Y = joint[:, :, 1]

        split[anno['train'][0][0]].append(v)
        bone_vid = []
        for f in range(0, nframes, strip):
            bone = get_skeleton(X[f], Y[f])
            bone_vid.append(bone)

        bone_vid = np.array(bone_vid, dtype=np.float)

        # vis('%04d' % v, bone_vid[:, :], nframes, act, strip=1)

        save_file = os.path.join(bone_dir, '%04d_cropPart.npz' % v)
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

def draw_skeleton(img, bones, w, h, m=60):
    for j in range(bones.shape[0]):
        x1, y1, x2, y2 = bones[j]
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)
        # cv2.rectangle(img, (x[j] - m, y[j] -m), (x[j] + m, y[j] + m), (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
    return


# 1.  head
# 2.  left_shoulder  3.  right_shoulder
# 4.  left_elbow     5.  right_elbow
# 6.  left_wrist     7.  right_wrist
# 8.  left_hip       9.  right_hip
# 10. left_knee      11. right_knee
# 12. left_ankle     13. right_ankle

# 0. body
# 1. left arm       2. right arm
# 3. left leg       4. right leg
def get_skeleton(x, y):
    bone = []
    m = 0.085

    # body
    l, r, t, b = bbox_from_pt(x, y, [0, 1, 2, 7, 8], m)
    bone.append([l, t, r, b])

    # left arm
    l, r, t, b = bbox_from_pt(x, y, [1, 3, 5], m)
    bone.append([l, t, r, b])
    # right arm
    l, r, t, b = bbox_from_pt(x, y, [2, 4, 6], m)
    bone.append([l, t, r, b])
    # left leg
    l, r, t, b = bbox_from_pt(x, y, [7, 9, 11], m)
    bone.append([l, t, r, b])
    l, r, t, b = bbox_from_pt(x, y, [8, 10, 12], m)
    bone.append([l, t, r, b])

    return bone


def bbox_from_pt(X, Y, jts, m):
    l = np.min(X[jts]) - m
    r = np.max(X[jts]) + m
    t = np.min(Y[jts]) - m
    b = np.max(Y[jts]) + m
    return l, r, t, b


if __name__ == '__main__':
    # enum()
    save_pose(strip=1)