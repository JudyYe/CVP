# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------


import torch
import torch.nn.functional as F
from utils.box_utils import logcenters_to_extents

"""
Functions for computing image layouts from object vectors, bounding boxes,
and segmentation masks. These are used to compute course scene layouts which
are then fed as input to the cascaded refinement network.
"""


def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [xc, yc, logw, logh] in the [0, 1] coordinate space
    - obj_to_img: LongTensor of shape (O,) mapping each element of vecs to
      an image, where each element is in the range [0, N). If obj_to_img[i] = j
      then vecs[i] belongs to image j.
    - H, W: Size of the output

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    O, D = vecs.size()
    if W is None:
        W = H

    extend_boxes = logcenters_to_extents(boxes)
    grid = _boxes_to_grid(extend_boxes, H, W)

    # If we don't add extra spatial dimensions here then out-of-bounds
    # elements won't be automatically set to 0
    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid)  # (O, D, H, W)

    # Explicitly masking makes everything quite a bit slower.
    # If we rely on implicit masking the interpolated boxes end up
    # blurred around the edges, but it should be fine.
    # mask = ((X < 0) + (X > 1) + (Y < 0) + (Y > 1)).clamp(max=1)
    # sampled[mask[:, None]] = 0

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out

def boxes_to_region_layout(obj_vecs, layout_boxes, RH, RW):
    """
    :param obj_vecs: (O, D)
    :param layout_boxes: (O, 4)
    :param RH, RW: size of output
    :return: (O, D, RH, RW)
    """
    O, D = obj_vecs.size()
    out = obj_vecs.view(O, D, 1, 1).expand([O, D, RH, RW])
    return out

def sg2im_masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space
    - masks: Tensor of shape (O, M, M) giving binary masks for each object
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - H, W: Size of the output image.

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid)

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)
    return out

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BACK_MASK = 1e-1
BIG_NEG = -1e+6

def mask_to_layout(bbox, mask, color, target_size):
    """
    independent mask
    :param bbox Tensor of shape (N, O, 4) [x0, y0, x1, y1] in [0, 1]
    :param mask: Tensor of shape(N, O, 1, h, w)
    :param color:Tensor of shape(N, O, 3, h, w)
    :param target_size: H, W size of the output image
    :return: Tensor of shape (N, 3, H, W)
    """

    H, W = target_size
    N, O, _, h, w = mask.size()

    back_color = -torch.FloatTensor(IMAGENET_MEAN) / torch.FloatTensor(IMAGENET_STD)
    back_color = back_color.view(1, 3, 1, 1).expand(1, 3, H, W).cuda()
    back_mask = torch.FloatTensor([BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()

    layout_list = []
    mask_list = []
    for n in range(N):
        grid = _boxes_to_grid(bbox[n], H, W)  # O, H, W, 2
        mask_fore = F.grid_sample(mask[n], grid)  # (O, 1, H, W)
        mask_fb = torch.cat([mask_fore, back_mask]) # (O + 1, 1, H, W)
        mask_fb = mask_norm(mask_fb, norm='p-10')  # (O + 1, 1, H, W)

        color_fore = F.grid_sample(color[n], grid, padding_mode='border')
        color_fb = torch.cat([color_fore, back_color], dim=0)

        # nan = torch.isnan(mask_fb).nonzero()
        # print('mask', nan)

        color_layout = color_fb * mask_fb
        layout = torch.sum(color_layout, dim=0)  # 3, H, W
        layout_list.append(layout)
        mask_list.append(mask_fb)
    layout_batch = torch.stack(layout_list)  # (N, 3, H, W)
    mask_fb = torch.stack(mask_list) # (N, O + 1, 1, H, W)
    return layout_batch, mask_fb


def mask_to_bg(bbox, mask, color, bg, target_size, V=None):
    """
    independent mask
    :param bbox Tensor of shape (N, O, 4) [x0, y0, x1, y1] in [0, 1]
    :param mask: Tensor of shape(N, O, 1, h, w)
    :param color:Tensor of shape(N, O, D, h, w)
    :param bg: Tenosr of shape (V, D, H, W)
    :param target_size: H, W size of the output image
    :return: Tensor of shape (N, D, H, W)
    """

    H, W = target_size
    N, O, _, h, w = mask.size()
    if V is None:
        V = bg.size(0)
    dt = N // V

    if bg is None:
        back_color = -torch.FloatTensor(IMAGENET_MEAN) / torch.FloatTensor(IMAGENET_STD)
        back_color = back_color.view(1, 3, 1, 1).expand(1, 3, H, W).cuda()
        back_mask = torch.FloatTensor([BACK_MASK * BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()
    else:
        back_color = bg
        back_mask = torch.FloatTensor([BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()

    layout_list = []
    mask_list = []
    for n in range(N):
        #  (dt, v) -> (N), dt, v = divmod(n, V)
        # v = n // dt
        v = divmod(n, V)[1]
        grid = _boxes_to_grid(bbox[n], H, W)  # O, H, W, 2
        mask_fore = F.grid_sample(mask[n], grid)  # (O, 1, H, W)
        mask_fb = torch.cat([mask_fore, back_mask]) # (O + 1, 1, H, W)
        mask_fb = mask_norm(mask_fb, norm='p-10')  # (O + 1, 1, H, W)

        color_fore = F.grid_sample(color[n], grid, padding_mode='border')
        color_fb = torch.cat([color_fore, back_color[v: v+1]], dim=0)

        # nan = torch.isnan(mask_fb).nonzero()
        # print('mask', nan)
        color_layout = color_fb * mask_fb
        layout = torch.sum(color_layout, dim=0)  # 3, H, W
        layout_list.append(layout)
        mask_list.append(mask_fb)
    layout_batch = torch.stack(layout_list)  # (N, 3, H, W)
    mask_fb = torch.stack(mask_list) # (N, O + 1, 1, H, W)
    return layout_batch, mask_fb

def bbox_to_bg_feat(bbox, color, bg, V=None):
    """
    independent mask
    :param bbox Tensor of shape (N, O, 4) [x0, y0, x1, y1] in [0, 1]
    :param color:Tensor of shape(N, O, D, h, w)
    :param bg: Tenosr of shape  (N, D, H, W)
    :param target_size: H, W size of the output image
    :return: Tensor of shape (N, D, H, W)
    """

    N, D, H, W = bg.size()
    assert color.size(2) == D
    if V is None:
        V = bg.size(0)

    if bg is None:
        back_color = -torch.FloatTensor(IMAGENET_MEAN) / torch.FloatTensor(IMAGENET_STD)
        back_color = back_color.view(1, 3, 1, 1).expand(1, 3, H, W).cuda()
        back_mask = torch.FloatTensor([BACK_MASK * BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()
    else:
        back_color = bg

    layout_list = []
    for n in range(N):
        #  (dt, v) -> (N), dt, v = divmod(n, V)
        # v = n // dt
        grid = _boxes_to_grid(bbox[n], H, W)  # O, H, W, 2
        color_fore = F.grid_sample(color[n], grid)  # (O, D, H, W)

        color_fb = torch.cat([color_fore, back_color[n:n+1]], dim=0) # (O + 1, D, H, W)
        layout = torch.max(color_fb, dim=0)[0] # (D, H, W)

        layout_list.append(layout)
    layout_batch = torch.stack(layout_list)  # (N, D, H, W)
    return layout_batch


def splat_to_bg_feat(bbox, color, bg, V=None):
    """
    stole from https://github.com/google/layered-scene-inference/blob/master/lsi/geometry/sampling.py#L171
    :param bbox Tensor of shape (N, O, 4) [x0, y0, x1, y1] in [0, 1]
    :param color:Tensor of shape(N, O, D, h, w)
    :param bg: Tenosr of shape  (N, D, H, W)
    :param target_size: H, W size of the output image
    :return: Tensor of shape (N, D, H, W)
    """

    N, D, H, W = bg.size()
    assert color.size(2) == D
    if V is None:
        V = bg.size(0)

    if bg is None:
        back_color = -torch.FloatTensor(IMAGENET_MEAN) / torch.FloatTensor(IMAGENET_STD)
        back_color = back_color.view(1, 3, 1, 1).expand(1, 3, H, W).cuda()
        back_mask = torch.FloatTensor([BACK_MASK * BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()
    else:
        back_color = bg

    layout_list = []
    for n in range(N):
        #  (dt, v) -> (N), dt, v = divmod(n, V)
        # v = n // dt
        color_fore = splat(bbox[n], color[n], H, W)  # (O, D, H, W)

        color_fb = torch.cat([color_fore, back_color[n:n+1]], dim=0) # (O + 1, D, H, W)
        layout = torch.max(color_fb, dim=0)[0] # (D, H, W)

        layout_list.append(layout)
    layout_batch = torch.stack(layout_list)  # (N, D, H, W)
    return layout_batch


def mask_splat_to_bg(bbox, mask, fg_feat, bg_feat):
    """
    :param bbox: (O, 4)
    :param mask: (N, O, 1, h, w)
    :param fg_feat: (N, O, D, h, w)
    :param bg_feat: (V, D, H, W)
    :return: (N, D, H, W)
    """
    V, D, H, W = bg_feat.size()
    N = fg_feat.size(0)
    # print('bg', bg_feat.size(), 'fg', fg_feat.size(), 'V', V, 'N', N)
    # assert False
    assert fg_feat.size(2) == D
    layout_list, mask_list = [], []
    back_mask = torch.FloatTensor([BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()
    # back_mask = torch.FloatTensor([BACK_MASK * BACK_MASK]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()

    for n in range(N):
        v = divmod(n, V)[1]
        color_fore = splat(bbox[n], fg_feat[n], H, W) # (O, D, H, W)
        color_fb = torch.cat([color_fore, bg_feat[v: v+1]], dim=0)

        mask_fore = splat(bbox[n], mask[n], H, W)
        mask_fb = torch.cat([mask_fore, back_mask]) # (O + 1, 1, H, W)
        mask_fb = mask_norm(mask_fb, norm='p-10')

        feat_layout = color_fb * mask_fb
        feat_layout = torch.sum(feat_layout, dim=0)
        layout_list.append(feat_layout)
        mask_list.append(mask_fb)
    feat_fb = torch.stack(layout_list)
    mask_fb = torch.stack(mask_list)
    return feat_fb, mask_fb


def soft_mask_splat_to_bg(bbox, mask, fg_feat, bg_feat):
    V, D, H, W = bg_feat.size()
    N = fg_feat.size(0)
    assert fg_feat.size(2) == D
    layout_list, mask_list = [], []
    back_mask = torch.FloatTensor([-10]).view(1, 1, 1, 1).expand(1, 1, H, W).cuda()

    for n in range(N):
        v = divmod(n, V)[1]
        color_fore = splat(bbox[n], fg_feat[n], H, W) # (O, D, H, W)
        color_fb = torch.cat([color_fore, bg_feat[v: v+1]], dim=0)

        mask_fore = splat_neg(bbox[n], mask[n], H, W)
        mask_fb = torch.cat([mask_fore, back_mask]) # (O + 1, 1, H, W)
        mask_fb = mask_norm(mask_fb, norm='softmax')

        feat_layout = color_fb * mask_fb
        feat_layout = torch.sum(feat_layout, dim=0)
        layout_list.append(feat_layout)
        mask_list.append(mask_fb)
    feat_fb = torch.stack(layout_list)
    mask_fb = torch.stack(mask_list)
    return feat_fb, mask_fb


def splat_with_wgt(bbox, feat, H, W):
    """
    :param bbox: (O, 4) in [0, 1]
    :param feat: (O, D, h, w)
    :return: (O, D, H, W)
    """
    N, D, h, w = feat.size()
    grid = _bbox_to_grid_fwd01(bbox, h, w)  # (N, h, w, 2)
    grid[:, :, :, 0] = grid[:, :, :, 0] * W
    grid[:, :, :, 1] = grid[:, :, :, 1] * H

    num_pixels_src = h * w
    num_pixels_trg = H * W
    grid = grid - 0.5
    x = grid[:, :, :, 0].view(N, 1, h, w)
    y = grid[:, :, :, 1].view(N, 1, h, w)
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    y_max = H - 1
    x_max = W - 1

    x0_safe = torch.clamp(x0, min=0, max=x_max)
    y0_safe = torch.clamp(y0, min=0, max=y_max)
    x1_safe = torch.clamp(x1, min=0, max=x_max)
    y1_safe = torch.clamp(y1, min=0, max=y_max)

    wt_x0 = (x1 - x) * torch.eq(x0, x0_safe).to(bbox)
    wt_x1 = (x - x0) * torch.eq(x1, x1_safe).to(bbox)
    wt_y0 = (y1 - y) * torch.eq(y0, y0_safe).to(bbox)
    wt_y1 = (y - y0) * torch.eq(y1, y1_safe).to(bbox)

    wt_tl = wt_x0 * wt_y0
    wt_tr = wt_x1 * wt_y0
    wt_bl = wt_x0 * wt_y1
    wt_br = wt_x1 * wt_y1

    eps = 1e-3
    wt_tl = torch.clamp(wt_tl, min=eps)
    wt_tr = torch.clamp(wt_tr, min=eps)
    wt_bl = torch.clamp(wt_bl, min=eps)
    wt_br = torch.clamp(wt_br, min=eps)

    values_tl = (feat * wt_tl).view(N, D, num_pixels_src)  # (N, D, h, w)
    values_tr = (feat * wt_tr).view(N, D, num_pixels_src)  # (N, D, h, w)
    values_bl = (feat * wt_bl).view(N, D, num_pixels_src)  # (N, D, h, w)
    values_br = (feat * wt_br).view(N, D, num_pixels_src)  # (N, D, h, w)

    inds_tl = (x0_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)
    inds_tr = (x1_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)
    inds_bl = (x0_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)
    inds_br = (x1_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, D, num_pixels_src)

    init_trg_image = torch.zeros([N, D, num_pixels_trg]).to(bbox)
    init_trg_image = init_trg_image.scatter_add(-1, inds_tl, values_tl)
    init_trg_image = init_trg_image.scatter_add(-1, inds_tr, values_tr)
    init_trg_image = init_trg_image.scatter_add(-1, inds_bl, values_bl)
    init_trg_image = init_trg_image.scatter_add(-1, inds_br, values_br)

    # cnt weight
    wt_tl = wt_tl.view(N, 1, num_pixels_src)
    wt_tr = wt_tr.view(N, 1, num_pixels_src)
    wt_bl = wt_bl.view(N, 1, num_pixels_src)
    wt_br = wt_br.view(N, 1, num_pixels_src)

    inds_tl = (x0_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)
    inds_tr = (x1_safe + y0_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)
    inds_bl = (x0_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)
    inds_br = (x1_safe + y1_safe * W).view(N, 1, num_pixels_src).long().expand(N, 1, num_pixels_src)

    init_trg_wgt = torch.zeros([N, 1, num_pixels_trg]).to(bbox) + eps
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tl, wt_tl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tr, wt_tr)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_bl, wt_bl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_br, wt_br)

    init_trg_image = init_trg_image.view(N, D, H, W)
    init_trg_wgt = init_trg_wgt.view(N, 1, H, W)
    init_trg_image = init_trg_image / init_trg_wgt

    init_trg_wgt = torch.zeros([N, 1, num_pixels_trg]).to(bbox)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tl, wt_tl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_tr, wt_tr)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_bl, wt_bl)
    init_trg_wgt = init_trg_wgt.scatter_add(-1, inds_br, wt_br)

    init_trg_wgt = init_trg_wgt.view(-1)
    inds = (init_trg_wgt == 0).nonzero()
    init_trg_wgt[inds] = BIG_NEG
    init_trg_wgt = torch.clamp(init_trg_wgt, max=0)
    init_trg_wgt = init_trg_wgt.view(N, 1, H, W)

    return init_trg_image, init_trg_wgt


def splat(bbox, feat, H, W):
    """
    :param bbox: (O, 4) in [0, 1]
    :param feat: (O, D, h, w)
    :return: (O, D, H, W)
    """
    init_trg_image, _ = splat_with_wgt(bbox, feat, H, W)

    return init_trg_image

def splat_neg(bbox, feat, H, W):
    image, wgt = splat_with_wgt(bbox, feat, H, W)
    image = image + wgt
    return image

def bbox_to_mask(bbox, target_size):
    """
    :param bbox: (N, O, 4)
    :param target_size: H, W
    :return: (N, 1, 1, H, W)
    """
    H, W = target_size
    N, O, _ = bbox.size()

    SOME_BIG = 8
    ones = torch.FloatTensor([1.]).view(1, 1, 1, 1).expand(O, 1, SOME_BIG, SOME_BIG).cuda()
    layout_list = []
    mask_list = []
    for n in range(N):
        grid = _boxes_to_grid(bbox[n], H, W)  # O, H, W, 2
        mask_fore = F.grid_sample(ones, grid)  # (O, 1, H, W)
        mask_fore = torch.sum(mask_fore, dim=0, keepdim=True) # (1, 1, H, W)
        mask_fore = torch.clamp(mask_fore, max=1)

        mask_list.append(mask_fore)
    mask = torch.stack(mask_list) # (N, 1, 1, H, W)
    return mask


def gauss_mask(bbox, gauss_kernel, gauss_size, target_size):
    """
    :param bbox: (N, O, 4)
    :param target_size: H, W
    :return: (N, O, 1, H, W)
    """
    H, W = target_size
    N, O, _ = bbox.size()

    ones = gauss_kernel.cuda().expand(O, 1, gauss_size, gauss_size)
    mask_list = []
    for n in range(N):
        grid = _boxes_to_grid(bbox[n], H, W)  # O, H, W, 2
        mask_fore = F.grid_sample(ones, grid)  # (O, 1, H, W)
        mask_list.append(mask_fore)
    mask = torch.stack(mask_list) # (N, O, 1, H, W)
    return mask


def bbox_to_bg(bbox, color, bg, target_size):
    """
    independent mask
    :param bbox Tensor of shape (N, O, 4) [x0, y0, x1, y1] in [0, 1]
    :param color:Tensor of shape(N, 1, 3, h, w), N = dt * V
    :param bg: Tenosr of shape (V, 3, H, W), V
    :param target_size: H, W size of the output image
    :return: Tensor of shape (N, 3, H, W)
    """

    H, W = target_size
    N, O, _ = bbox.size()
    V = bg.size(0)
    dt = N // V

    SOME_BIG = 8
    ones = torch.FloatTensor([1.]).view(1, 1, 1, 1).expand(O, 1, SOME_BIG, SOME_BIG).cuda()
    layout_list = []
    mask_list = []
    for n in range(N):
        # v = n // dt
        v = divmod(n, V)[1]
        grid = _boxes_to_grid(bbox[n], H, W)  # O, H, W, 2
        mask_fore = F.grid_sample(ones, grid)  # (O, 1, H, W)

        mask_fore = torch.sum(mask_fore, dim=0, keepdim=True) # (1, 1, H, W)
        mask_fore = torch.clamp(mask_fore, max=1)
        mask_fb = torch.cat([mask_fore, 1 - mask_fore]) # (2, 1, H, W)

        # color_fore = F.grid_sample(color[n], grid, padding_mode='border')
        color_fb = torch.cat([color[n], bg[v: v+1]], dim=0) # (2, 3, H, W))
        color_layout = color_fb * mask_fb
        layout = torch.sum(color_layout, dim=0)  # 3, H, W
        layout_list.append(layout)
        mask_list.append(mask_fb)
    layout_batch = torch.stack(layout_list)  # (N, 3, H, W)
    mask_fb = torch.stack(mask_list) # (N, 1 + 1, 1, H, W)
    return layout_batch, mask_fb

def mask_to_region(bbox, mask, color):
    """
    :param bbox:  Tensor of shape (N, O, 2) [xc, yc]
    :param mask:  (N, O+1, 1, H, W)
    :param color: (N, O, C, h, w)
    :param target_size:
    :return: Tensor of shape (N, o, C, h, w)
    """
    N, O, C, h, w = color.size()
    region_list = []
    mask_list = []
    back_color = -torch.FloatTensor(IMAGENET_MEAN) / torch.FloatTensor(IMAGENET_STD)
    back_color = back_color.view(1, 3, 1, 1).expand(1, 3, h, w).cuda()
    for n in range(N):
        grid = _boxes_to_grid_inv(bbox[n], h, w) # (O, h, w, 2) #todo
        mask_region = F.grid_sample(mask[n][0:O], grid) # (O, 1, h, w)
        region = color[n] * mask_region + back_color * (1 - mask_region)
        region_list.append(region)
        mask_list.append(mask_region)
    region = torch.stack(region_list)  # (N, O, C, h, w)
    mask = torch.stack(mask_list)
    return region, mask


def bbox_to_region(bbox, color, h, w):
    """
    :param bbox:  Tensor of shape (N, O, 2) [xc, yc]
    :param mask:  (N, 1+1, 1, H, W)
    :param color: (N, 1, C, h, w)
    :param target_size:
    :return: Tensor of shape (N, o, C, h, w)
    """
    N, O, C, H, W = color.size()
    region_list = []
    for n in range(N):
        grid = _boxes_to_grid_inv(bbox[n], h, w) # (O, h, w, 2) #todo
        region = F.grid_sample(color[n][0:O], grid)
        region_list.append(region)
    region = torch.stack(region_list)  # (N, O, C, h, w)
    return region, None

def mask_norm(logits, norm='softmax', eps=1e-7):
    """
    :param logit: (O, 1, H, W)
    :return: softmax(O, 1, H, W)
    """
    # if all of channel > 0: softmax
    # if some of channel > 0:
    if norm == 'softmax':
        normed_tensor = torch.nn.functional.softmax(logits, dim=0)
    elif norm[0] == 'p':
        p = int(norm.split('-')[-1])
        # normed_tensor = (logits.pow(p) + eps) / (torch.sum(logits.pow(p), dim=0, keepdim=True) + eps)
        normed_tensor = logits.pow(p) / (torch.sum(logits.pow(p), dim=0, keepdim=True))
    else:
        raise NotImplementedError
    return normed_tensor


def _bbox_to_grid_fwd01(boxes, H, W):
    """
    :param boxes: (O, 4)  [x0, y0, x1, y1] in (0, 1)
    :return: (O, H, W, 2)
    """
    O = boxes.size(0)
    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)
    x0 = boxes[:, 0].view(O, 1, 1)
    y0 = boxes[:, 1].view(O, 1, 1)
    x1 = boxes[:, 2].view(O, 1, 1)
    y1 = boxes[:, 3].view(O, 1, 1)
    X = X * (x1 - x0) + x0
    Y = Y * (y1 - y0) + y0
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)
    return grid


def _boxes_to_grid_inv(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1] in (0, 1)
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output

    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)
    x0 = boxes[:, 0].view(O, 1, 1).expand(O, 1, W)
    y0 = boxes[:, 1].view(O, 1, 1).expand(O, H, 1)
    x1 = boxes[:, 2].view(O, 1, 1).expand(O, 1, W)
    y1 = boxes[:, 3].view(O, 1, 1).expand(O, H, 1)

    # X = torch.linspace(x0, x1, steps=W).view(1, 1, W).to(boxes)
    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    # Y = torch.linspace(y0, y1, steps=H).view(1, H, 1).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = X.expand(O, 1,  W)
    X = X * (x1 - x0) + x0 # (x0, x1)
    X = X.expand(O, H, W)

    Y = Y.expand(O, H, 1)
    Y = Y * (y1 - y0) + y0 # [y0, y1]
    Y = Y.expand(O, H, W)

    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)
    return grid

def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output

    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def _pool_samples(samples, obj_to_img, pooling='sum'):
    """
    Input:
    - samples: FloatTensor of shape (O, D, H, W)
    - obj_to_img: LongTensor of shape (O,) with each element in the range
      [0, N) mapping elements of samples to output images

    Output:
    - pooled: FloatTensor of shape (N, D, H, W)
    """
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    # Use scatter_add to sum the sampled outputs for each image
    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    if pooling == 'avg':
        # Divide each output mask by the number of objects; use scatter_add again
        # to count the number of objects per image.
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out


if __name__ == '__main__':
    vecs = torch.FloatTensor([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ])
    boxes = torch.FloatTensor([
        [0.25, 0.125, 0.5, 0.875],
        [0, 0, 1, 0.25],
        [0.6125, 0, 0.875, 1],
        [0, 0.8, 1, 1.0],
        [0.25, 0.125, 0.5, 0.875],
        [0.6125, 0, 0.875, 1],
    ])
    obj_to_img = torch.LongTensor([0, 0, 0, 1, 1, 1])
    # vecs = torch.FloatTensor([[[1]]])
    # boxes = torch.FloatTensor([[[0.25, 0.25, 0.75, 0.75]]])
    vecs, boxes = vecs.cuda(), boxes.cuda()
    obj_to_img = obj_to_img.cuda()
    out = boxes_to_layout(vecs, boxes, obj_to_img, 256, pooling='sum')

    from torchvision.utils import save_image

    save_image(out.data, 'out.png')

    masks = torch.FloatTensor([
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    ])
    masks = masks.cuda()
    out = masks_to_layout(vecs, boxes, masks, obj_to_img, 256)
    save_image(out.data, 'out_masks.png')
