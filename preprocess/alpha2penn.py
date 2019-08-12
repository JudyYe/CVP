# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import cv2

import numpy as np
import os
import json

# valid = bone_vid['valid']

#
# fexp = '/scratch/yufeiy2/Penn_Action/bones/0801_Joint.npz'
# bone_vid = np.load(fexp)
# print(bone_vid['valid']) # (13, )
# print(bone_vid['bone'].shape) # dt, 13, 2 in [0, 1]


alpha_file = '/nfs.yoda/xiaolonw/judy_folder/transfer/penn_key/alphapose-results.json'
img_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/penn_vis/'
save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/penn_key/img/'

def sep_cvt_npz():
    with open(alpha_file) as fp:
        all_dets = json.load(fp)
    fname = {}
    for i, each_person in enumerate(all_dets):
        if i % 100 == 0:
            print('%d / %d' % (i, len(all_dets)))
        draw_mpi(each_person)
        penn = cvt_mpi2penn(each_person)
        index = penn['image_id'].split('.')[0]
        if index not in fname:
            fname[index] = {'bones': [], 'score': []}
        fname[index]['bones'].append(penn['keypoints'])
        fname[index]['score'].append(penn['score'])
        # draw_penn(penn)

        if i > 100 :
            break
    for index in fname:
        mul_person = fname[index]
        score = np.array(mul_person['score'])
        idx = np.argsort(-score)
        mul_person['bones'] = np.array(mul_person['bones'])[idx]

        save_file = os.path.join(save_dir, index + '_m13.npz')
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.savez_compressed(save_file, bones=mul_person['bones'], score=mul_person['score'])
    


def cvt_mpi2penn(person):
    # 0.  head
    # 1.  left_shoulder  2.  right_shoulder
    # 3.  left_elbow     4.  right_elbow
    # 5.  left_wrist     6.  right_wrist
    # 7.  left_hip       8.  right_hip
    # 9. left_knee       10. right_knee
    # 11. left_ankle     12. right_ankle

    # {0, "Nose"},
    # {1, "LEye"},
    # {2, "REye"},
    # {3, "LEar"},
    # {4, "REar"},
    # {5, "LShoulder"},
    # {6, "RShoulder"},
    # {7, "LElbow"},
    # {8, "RElbow"},
    # {9, "LWrist"},
    # {10, "RWrist"},
    # {11, "LHip"},
    # {12, "RHip"},
    # {13, "LKnee"},
    # {14, "Rknee"},
    # {15, "LAnkle"},
    # {16, "RAnkle"},
    src = np.array(person['keypoints']).reshape([17, 3])
    dst = np.zeros([13, 3])
    dst[0] = src[0]
    dst[1:] = src[5:]
    person['keypoints'] = dst
    return person


def draw_mpi(person):
    index = person['image_id'].split('.')[0]
    img = cv2.imread(os.path.join(img_dir, person['image_id']))
    bones = np.array(person['keypoints']).reshape([17, 3])
    box = bones.astype(np.int64)
    for i in range(17):
        cv2.circle(img, (box[i][0], box[i][1]), 5, (255, 255, 255), -1)
    save_file = os.path.join(save_dir, index + '_mpi.jpg')
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    cv2.imwrite(save_file, img)

def draw_penn(person):
    index = person['image_id'].split('.')[0]
    img = cv2.imread(os.path.join(img_dir, person['image_id']))
    box = np.array(person['keypoints']).reshape([13, 3])
    box = box[:, 0:2].astype(np.int)

    for i in range(13):
        cv2.circle(img, (box[i][0], box[i][1]), 5, (255, 255, 255), -1)

    cv2.line(img, tuple(box[0]), tuple(box[1]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[0]), tuple(box[2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[1]), tuple(box[2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[1]), tuple(box[3]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[2]), tuple(box[4]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[3]), tuple(box[5]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[4]), tuple(box[6]), (255, 255, 255), 2)

    cv2.line(img, tuple(box[1]), tuple(box[7]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[2]), tuple(box[8]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[7]), tuple(box[8]), (255, 255, 255), 2)

    cv2.line(img, tuple(box[7]), tuple(box[9]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[9]), tuple(box[11]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[8]), tuple(box[10]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[10]), tuple(box[12]), (255, 255, 255), 2)

    save_file = os.path.join(save_dir, index + '_penn.jpg')
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    cv2.imwrite(save_file, img)




if __name__ == '__main__':
    sep_cvt_npz()

    # [
    # //
    # for person_1 in image_1
    # {
    # "image_id": string, image_1_name,
    #             "category_id": int, 1
    # for person
    # "keypoints": [x1, y1, c1, ..., xk, yk, ck],
    #              "score": float,
    # },
    # // for person_2 in image_1
    # {
    # "image_id": string, image_1_name,
    #             "category_id": int, 1
    # for person
    # "keypoints": [x1, y1, c1, ..., xk, yk, ck],
    #              "score": float,
    # },
    # ...
    # //
    # for persons in image_2
    # {
    # "image_id": string, image_2_name,
    #             "category_id": int, 1
    # for person
    # "keypoints": [x1, y1, c1, ..., xk, yk, ck],
    #              "score": float,
    # },
    # ...
    # ]