#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np

"""
Utilities for dealing with bounding boxes (in pytorch)
"""

def apply_box_transform(anchors, transforms):
    """
    Apply box transforms to a set of anchor boxes.

    Inputs:
    - anchors: Anchor boxes of shape (N, 4), where each anchor is specified
      in the form [xc, yc, w, h]
    - transforms: Box transforms of shape (N, 4) where each transform is
      specified as [tx, ty, tw, th]

    Returns:
    - boxes: Transformed boxes of shape (N, 4) where each box is in the
      format [xc, yc, w, h]
    """
    # Unpack anchors
    xa, ya = anchors[:, 0], anchors[:, 1]
    wa, ha = anchors[:, 2], anchors[:, 3]

    # Unpack transforms
    tx, ty = transforms[:, 0], transforms[:, 1]
    tw, th = transforms[:, 2], transforms[:, 3]

    x = xa + tx * wa
    y = ya + ty * ha

    if torch.is_tensor(anchors):
        w = wa * tw.exp()
        h = ha * th.exp()

        boxes = torch.stack([x, y, w, h], dim=1)
    else:
        w = wa * np.exp(tw)
        h = ha * np.exp(th)

        boxes = np.vstack((x, y, w, h)).transpose()

    return boxes


def transform_to_xyxy(anchors, transform):
    """
    :param anchors: (N, 4) in for (x1, y1, x2, y2)
    :param transform:
    :return: (x1, y1, x2, y2)
    """
    anchors = extents_to_centers(anchors)
    boxes = apply_box_transform(anchors, transform)
    boxes = centers_to_extents(boxes)
    return boxes


def invert_box_transform(anchors, boxes):
    """
    Compute the box transform that, when applied to anchors, would give boxes.

    Inputs:
    - anchors: Box anchors of shape (N, 4) in the format [xc, yc, w, h]
    - boxes: Target boxes of shape (N, 4) in the format [xc, yc, w, h]

    Returns:
    - transforms: Box transforms of shape (N, 4) in the format [tx, ty, tw, th]
    """
    # Unpack anchors
    xa, ya = anchors[:, 0], anchors[:, 1]
    wa, ha = anchors[:, 2], anchors[:, 3]

    # Unpack boxes
    x, y = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2], boxes[:, 3]

    tx = (x - xa) / wa
    ty = (y - ya) / ha

    if torch.is_tensor(anchors):
        tw = w.log() - wa.log()
        th = h.log() - ha.log()

        transforms = torch.stack([tx, ty, tw, th], dim=1)
    else:
        tw = np.log(w) - np.log(wa)
        th = np.log(h) - np.log(ha)

        transforms = np.vstack((tx, ty, tw, th)).transpose()

    return transforms


def xyxy_to_transform(anchors, boxes):
    anchors = extents_to_centers(anchors)
    boxes = extents_to_centers(boxes)
    transform = invert_box_transform(anchors, boxes)
    return transform


def xy_to_xyxy(boxes, w):
    """
    :param boxes: Tensor of shape (N, 2) between [0,1]
    :param w: radius
    :return: Tensor of shape (N, 4) (x0y0 x1, y1)
    """

    xc = boxes[:, 0]
    yc = boxes[:, 1]
    x0 = xc - w
    y0 = yc - w
    x1 = xc + w
    y1 = yc + w
    box_out = box_stack((x0, y0, x1, y1))
    return box_out

def centers_to_extents(boxes):
    """
    Convert boxes from [xc, yc, w, h] format to [x0, y0, x1, y1] format

    Input:
    - boxes: Input boxes of shape (N, 4) in [xc, yc, w, h] format

    Returns:
    - boxes: Output boxes of shape (N, 4) in [x0, y0, x1, y1] format
    """
    xc, yc = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2], boxes[:, 3]

    x0 = xc - w / 2
    x1 = x0 + w
    y0 = yc - h / 2
    y1 = y0 + h
    boxes_out = box_stack((x0, y0, x1, y1))
    return boxes_out


def extents_to_centers(boxes):
    """
    Convert boxes from [x0, y0, x1, y1] format to [xc, yc, w, h] format

    Input:
    - boxes: Input boxes of shape (N, 4) in [x0, y0, x1, y1] format

    Returns:
    - boxes: Output boxes of shape (N, 4) in [xc, yc, w, h] format
    """
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]

    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0
    boxes_out = box_stack([xc, yc, w, h])
    # if torch.is_tensor(boxes):
    #     boxes_out = torch.stack([xc, yc, w, h], dim=1)
    # else:
    #     boxes_out = np.vstack((xc, yc, w, h)).transpose()
    return boxes_out


def logcenters_to_extents(boxes):
    # todo: check
    xc, yc = boxes[:, 0], boxes[:, 1]
    if torch.is_tensor(boxes):
        w, h = boxes[:, 2].exp(), boxes[:, 3].exp()  # (0, 1)
    else:
        w, h = np.exp(boxes[:, 2]), np.exp(boxes[:, 3])

    x0 = xc - w / 2
    x1 = x0 + w
    y0 = yc - h / 2
    y1 = y0 + h
    boxes_out = box_stack([x0, y0, x1, y1])
    # if torch.is_tensor(boxes):
    #     boxes_out = torch.stack([x0, y0, x1, y1], dim=1)
    # else:
    #     boxes_out = np.hstack((x0[:, np.newaxis], y0[:, np.newaxis], x1[:, np.newaxis], y1[:, np.newaxis]))
    return boxes_out


def extents_to_logcenters(boxes):
    # todo: check!!
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]

    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0

    if torch.is_tensor(boxes):
        w = w.log()
        h = h.log()
        boxes_out = torch.stack([xc, yc, w, h], dim=1)
    else:
        w = np.log(w)
        h = np.log(h)
        boxes_out = np.hstack((xc[:, np.newaxis], yc[:, np.newaxis], w[:, np.newaxis], h[:, np.newaxis]))
    return boxes_out

def extents_to_xy(boxes):
    xc = 0.5 * (boxes[:, 0] + boxes[:, 2])
    yc = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return np.hstack((xc[:, np.newaxis], yc[:, np.newaxis]))


def box_stack(input):
    if torch.is_tensor(input[0]):
        out = torch.stack(input, dim=1)
    else:
        out = np.vstack(input).transpose()
    return out

