# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import cv2
import torch
import numpy as np
from utils.data_utils import imagenet_deprocess_batch
from utils import box_utils
from . import layout as  layout_utils

"""
Utilities for making visualizations. stole some function from https://github.com/google/sg2im
"""

def convert_batch2cv(batch, max_num=1):
    """
    :param batch: (dt, V, 1, C, H, W)
    :return: list of image of video #0 in len dt
    """
    dt, V, O, C, H, W = batch.size()
    img_list = []
    v = 0
    img_numpy = imagenet_deprocess_batch(batch[:, v, 0].view(dt, C, H, W))
    for t in range(dt):
        img_list.append(convert_torch2cv(img_numpy[t]))
    return img_list


def convert_cv2torch(img_list):
    """
    :param img_list: list of [numpy in shape of (H, W, C)] in BGR
    :return: list of FloatTensor [(C, H, W)] in RGB
    """
    for i, img in enumerate(img_list):
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img_list[i] = torch.FloatTensor(img)
    return img_list


def convert_torch2cv(img):
    """
    :param img: FloatTensor in shape (C, H, W) in RGB
    :return: uint8 numpy in shape (H, W, C) in BGR
    """
    img = img.detach().numpy()
    img = img.transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_skeleton_color(pred_bboxes, dt, max_num=1, is_torch=True, O=3, valid=None, canvas=None, trip=None):
    V = pred_bboxes.size(1)
    O = pred_bboxes.size(2)
    if canvas is None:
        W = 128
    else:
        H, W = canvas.shape[0:2]

    dt = [0, 2, 5, -1]
    color = [(127, 127, 127), (0, 255, 255), (0, 127, 255), (0, 0 , 255)]
    V_idx = range(V)
    image_list = []
    pred_bboxes = pred_bboxes.cpu().detach().numpy() * W
    if trip is not None:
        trip = trip.cpu().detach().numpy().astype(np.int)
    for v in V_idx:
        for i in range(len(dt)):
            t = dt[i]
            pred_img = np.zeros([W, W, 3], dtype=np.float32)
            if trip is None:
                if valid is None:
                    if O == 13:
                        pred_img = skeleton_13B(pred_bboxes[t, v], pred_img)
                    elif O == 10:
                        pred_img = skeleton_10B(pred_bboxes[t, v], pred_img)
                else:
                    pred_img = skeleton_13B_valid(pred_bboxes[t, v], pred_img, valid)
            else:
                pred_img = skeleton_13B_trip(pred_bboxes[t, v], pred_img, trip[0, v], color[i])
            img = pred_img
            image_list.append(img)
        if v > max_num:
            break
    if is_torch:
        image_list = convert_cv2torch(image_list)
    return image_list


def get_skeleton_pred(pred_bboxes, dt, max_num=1, is_torch=True, O=3, valid=None, canvas=None, trip=None):
    V = pred_bboxes.size(1)
    O = pred_bboxes.size(2)
    if canvas is None:
        W = 128
    else:
        H, W = canvas.shape[0:2]


    V_idx = range(V)
    image_list = []
    pred_bboxes = pred_bboxes.cpu().detach().numpy() * W
    if trip is not None:
        trip = trip.cpu().detach().numpy().astype(np.int)
    for v in V_idx:
        for t in range(dt):
            # if canvas is None:
            #     pred_img = np.zeros([W, W, 3], dtype=np.float32)
            # else:
            #     pred_img = canvas
            pred_img = np.zeros([W, W, 3], dtype=np.float32)
            if trip is None:
                if valid is None:
                    if O == 13:
                        pred_img = skeleton_13B(pred_bboxes[t, v], pred_img)
                    elif O == 10:
                        pred_img = skeleton_10B(pred_bboxes[t, v], pred_img)
                else:
                    pred_img = skeleton_13B_valid(pred_bboxes[t, v], pred_img, valid)
            else:
                pred_img = skeleton_13B_trip(pred_bboxes[t, v], pred_img, trip[0, v])
            img = pred_img
            image_list.append(img)
        if v > max_num:
            break
    if is_torch:
        image_list = convert_cv2torch(image_list)
    return image_list


def skeleton_13B_trip(box, img, trip, color=(255, 255, 255)):
    """
    :param box: (O, 2)
    :param img:
    :param trip: (T, 3)
    :return:
    """
    for i in range(len(box)):
        cv2.circle(img, (box[i, 0], box[i, 1]), 5, color, -1)
    for t in range(len(trip)):
        start_idx = trip[t, 0]
        end_idx = trip[t, 2]
        cv2.line(img, tuple(box[start_idx]), tuple(box[end_idx]), color, 2)
    return img


def skeleton_13B_valid(box, img, valid):
    # colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
    #           (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors = [(255, 255, 255)] * 13
    bone = {0: [1, 2],
            1: [0, 2, 3, 7],
            2: [0, 1, 4, 8],
            3: [1, 5],
            4: [2, 6],
            5: [3],
            6: [4],
            7: [8, 1, 9],
            8: [7, 2, 10],
            9: [7, 11],
            10: [8, 12],
            11: [9],
            12: [10]}
    box = box.astype(np.int)
    for v, idx in enumerate(valid):
        if idx >= 0:
            cv2.circle(img, (box[idx, 0], box[idx, 1]), 5, colors[v], -1)
    for i in bone:
        if valid[i] < 0:
            continue
        for j in bone[i]:
            start_idx = valid[i]
            end_idx = valid[j]
            if valid[end_idx] < 0:
                continue
            cv2.line(img, tuple(box[start_idx]), tuple(box[end_idx]), (255, 255, 255), 2)
    return img

# 0.  head
# 1.  left_shoulder  2.  right_shoulder
# 3.  left_elbow     4.  right_elbow
# 5.  left_wrist     6.  right_wrist
# 7.  left_hip       8.  right_hip
# 9. left_knee       10. right_knee
# 11. left_ankle     12. right_ankle
def skeleton_13B(box, img):
    box = box.astype(np.int)
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
              (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for o in range(13):
        cv2.circle(img, (box[o, 0], box[o, 1]), 3, colors[o])

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
    return img

# 0: head, 1: body
# 2: leftUpperArm 3: rightUpperArm
# 4: leftLowerArm 5: rightLowerArm
# 6: leftUpperLeg 7: rightUpperLeg
# 8: leftLowerLeg 9: rightLowerLeg
def skeleton_10B(box, img):
    box = box.astype(np.int)
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
              (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for o in range(10):
        cv2.circle(img, (box[o, 0], box[o, 1]), 3, colors[o])

    cv2.line(img, tuple(box[0, 0: 2]), tuple(box[1, 0: 2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[1, 0: 2]), tuple(box[2, 0: 2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[1, 0: 2]), tuple(box[3, 0: 2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[2, 0: 2]), tuple(box[4, 0: 2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[3, 0: 2]), tuple(box[5, 0: 2]), (255, 255, 255), 2)

    cv2.line(img, tuple(box[1, 0: 2]), tuple(box[6, 0: 2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[1, 0: 2]), tuple(box[7, 0: 2]), (255, 255, 255), 2)

    cv2.line(img, tuple(box[6, 0: 2]), tuple(box[8, 0: 2]), (255, 255, 255), 2)
    cv2.line(img, tuple(box[7, 0: 2]), tuple(box[9, 0: 2]), (255, 255, 255), 2)
    return img


def get_bbox_traj(gt_bboxes, pred_bboxes, dt, max_num, is_torch=True, O=3, canvas=None):
    V = gt_bboxes.size(1)
    O = pred_bboxes.size(2)
    if canvas is None:
        W = 128
    else:
        H, W = canvas.shape[0:2]
    decay = 0.95
    V_idx = range(V)
    image_list = []
    gt_bboxes = gt_bboxes.cpu().detach().numpy() * W
    pred_bboxes = pred_bboxes.cpu().detach().numpy() * W
    # color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    color = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]
    for v in V_idx:
        gt_img = np.zeros([W, W, 3], dtype=np.float32)
        pred_img = np.zeros([W, W, 3], dtype=np.float32)
        for t in range(dt):
            for o in range(O):
                box = gt_bboxes[t, v, o].astype(np.int)
                if box[0] < 0 or box[0] > W or box[1] < 0 or box[1] > W:
                    continue
                cv2.circle(gt_img, (box[0], box[1]), radius=3, color=color[divmod(o, 3)[1]], thickness=-1)

                box = pred_bboxes[t, v, o].astype(np.int)
                if box[0] < 0 or box[0] > W or box[1] < 0 or box[1] > W:
                    continue
                cv2.circle(pred_img, (box[0], box[1]), radius=3, color=color[divmod(o, 3)[1]],thickness=-1)
            gt_img = gt_img * decay
            pred_img = pred_img * decay
        image_list.append(gt_img)
        image_list.append(pred_img)
        if v > max_num:
            break
    if is_torch:
        image_list = convert_cv2torch(image_list)
    return image_list


# todo: messy logic get bbox traj and get bbox traj image...
def get_bbox_traj_image(pred_bboxes, dt, max_num=2, is_torch=True, O=3, canvas=None, decay=0.9):
    V = pred_bboxes.size(1)
    O = pred_bboxes.size(2)
    image_list = []
    if canvas is None:
        W = 128
        pred_bboxes = pred_bboxes.cpu().detach().numpy() * W
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        # color = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]
    else:
        H, W = canvas.shape[0:2]
        pred_bboxes = pred_bboxes.cpu().detach().numpy() * W
        box = pred_bboxes[0, 0].astype(np.int) # (O, 2)
        box[:, 0] = np.maximum(np.minimum(box[:, 0], H - 1), 0)
        box[:, 1] = np.maximum(np.minimum(box[:, 1], W - 1), 0)
        colors = canvas[box[:, 1], box[:, 0]].astype(np.int).tolist()
        colors = [tuple(c) for c in colors]

    pred_img = np.zeros([W, W, 3], dtype=np.float32)
    for t in range(dt):
        for o in range(O):
            box = pred_bboxes[t, 0, o].astype(np.int)
            color = colors[divmod(o, 3)[1]] if o > len(colors) else colors[o]
            cv2.circle(pred_img, (box[0], box[1]), radius=3, color=color, thickness=-1)
        pred_img = pred_img * decay
    pred_img += canvas * 0.5
    image_list.append(pred_img)
    if is_torch:
        image_list = convert_cv2torch(image_list)
    return image_list


def draw_bbox_image(pred_bboxes, gt, radius=35):
    """
    :param pred_bboxes:
    :param gt: [(H, W, 3)]
    :param radius:
    :return:
    """
    dt, V, O, _ = pred_bboxes.size()
    H, W = gt[0].shape[0:2]

    pred_bboxes = pred_bboxes.cpu().detach().numpy() * W
    images = []
    for t in range(dt):
        box = pred_bboxes[t, 0].astype(np.int)  # (O, 2)
        box[:, 0] = np.maximum(np.minimum(box[:, 0], H - 1), 0)
        box[:, 1] = np.maximum(np.minimum(box[:, 1], W - 1), 0)
        img = gt[t].copy()
        for o in range(O):
            cv2.rectangle(img, (box[o, 0] - radius, box[o, 1] - radius), (box[o, 0] + radius, box[o, 1] + radius),
                          (255, 255, 255), 1)
        images.append(img)
    return images


def get_crop(pred, max_num=1, pool=1):
    if pool > 1:
        with torch.no_grad():
            dt, V, O, C, H, W = pred.size()
            pred = torch.nn.functional.avg_pool2d(pred.view(-1, pred.size(-1), pred.size(-1)), kernel_size=pool, stride=pool)
            pred = pred.view(dt, V, O, C, H // pool, W // pool)
    # dt, V, o, C, H, W
    dt, V, O = pred.size(0), pred.size(1), pred.size(2)
    images = []
    for n in range(V):
        for t in range(dt):
            pred_imgs = imagenet_deprocess_batch(pred[t, n])
            for o in range(O):
                images.append(pred_imgs[o])
        if n > max_num:
            break
    return images

def deprocess_dt_v_o_image(image):
    """
    :param image: Shape in (dt, V, o, C, H, W)
    :return: (dt, V, o, C, H, W), in range [0, 255]
    """
    dt, V, o, C, H, W = image.size()
    image = imagenet_deprocess_batch(image.view(dt * V * o, C, H, W))
    image = image.view(dt, V, o, C, H, W)
    return image



def get_layout_list(pred_bboxes, pred_region, num_step, radius, max_num=1, is_torch=True):
    """
    :param pred_bboxes:
    :param pred_region:
    :param num_step:
    :param radius:
    :param max_num:
    :param is_torch:
    :return: is_torch = false -> numpy [cv2]
    """
    # dt, v, o,
    image_list = []
    for t in range(num_step):
        pred_imgs = imagenet_deprocess_batch(pred_region[t, 0])
        # print(pred_imgs[0][:, 0, 0])
        image_list.append(pred_imgs[0])
    if not is_torch:
        image_list = [convert_torch2cv(each) for each in image_list]
        # print(image_list[0][0, 0])
    return image_list


def mask(region, bbox, num_step, radius, is_torch=True):
    """
    :param region: (dt, V, 1, C, H, W)
    :param bbox: (dt, V, o, D)
    :param num_step:
    :param radius:
    :param is_torch:
    :return:
    """
    dt, V, O, d = bbox.size()
    image_list = []
    if bbox.size(-1) == 2:
        xyxy_bbox = box_utils.xy_to_xyxy(bbox.view(dt* V * O, d), radius).view(dt * V, O, 4)
    elif bbox.size(-1) == 4:
        xyxy_bbox = box_utils.centers_to_extents(bbox.view(dt * V * O, d)).view(dt * V, O, 4)
    H, W = region.size(-1), region.size(-2)
    mask = layout_utils.bbox_to_mask(xyxy_bbox, (H, W)).view(dt, V, 1, 1, H, W)
    eps = 0.1
    mask = torch.clamp(mask + eps, max=1)
    pred_region = (region * mask).view(dt, 3, H, W)
    pred_imgs = imagenet_deprocess_batch(pred_region)
    for t in range(num_step):
        image_list.append(pred_imgs[t])
    if not is_torch:
        image_list = [convert_torch2cv(each) for each in image_list]
    return image_list


def render_pose(norm_box, valid, H, W):
    """
    :param box: (o, 2)
    :param valid:
    :return: (C, H, W)
    """
    bone = [(0, 1), (0 ,2),
            (1, 2), (7, 8),
            (1, 3), (1, 7),
            (2, 4), (2, 8),
            (3, 5), (4, 6),
            (7, 9), (8, 10),
            (9, 11), (10, 12),
            ]
    colors = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85));
    box = norm_box * W
    if torch.is_tensor(box):
        box = box.cpu().detach().numpy()
    box = box.astype(np.int)
    pose_canvas = np.zeros([H, W, 3])
    width = 2 * H // 64 + 1
    for v, idx in enumerate(valid):
        if idx > 0:
            cv2.circle(pose_canvas, (int(box[idx, 0]), int(box[idx, 1])), width, colors[v], -1)
    for i in range(len(bone)):
        src, dst = bone[i]
        if valid[src] <= 0:
            continue
        if valid[dst] <= 0:
            continue
        cv2.line(pose_canvas, (int(box[src, 0]), int(box[src, 1])), (int(box[dst, 0]), int(box[dst, 1])), colors[i], width)
    pose_canvas = pose_canvas.transpose([2, 0, 1])
    pose_canvas = pose_canvas / 255
    return torch.FloatTensor(pose_canvas)
