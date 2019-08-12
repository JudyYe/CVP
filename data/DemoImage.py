# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob
import os

import numpy as np
# import debug_init_paths
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import imagenet_preprocess, Resize


class DemoShapeStacks(Dataset):
    def __init__(self, image_dir, list_path, dt, radius, mod,
                 normalize_images=True, max_samples=None, training=False):
        super().__init__()

        self.RW = self.RH = self.W = self.H = 224
        self.orig_W = self.orig_H = 224
        self.box_rad = radius
        self.dt = 1

        self.image_dir = image_dir
        self.ext = '.jpg'
        self.max_samples = max_samples
        self.training = training
        self.modality = mod

        transform = [Resize((self.H, self.W)), T.ToTensor()]
        obj_transform = [Resize((self.RH, self.RW)), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
            obj_transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.obj_transform = T.Compose(obj_transform)
        with open(list_path) as fp:
            self.index_list = [line.strip() for line in fp]
        self.roidb = self.parse_gt_roidb()

    def parse_gt_roidb(self):
        roidb = {}
        for index in self.index_list:
            gt_path = os.path.join(self.image_dir, index + '.npy')
            bbox = np.load(gt_path) ## 3, 2 in (0, 224) coor
            roidb[index] = bbox
        return roidb

    def __len__(self):
        num = len(self.index_list)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def __getitem__(self, index):
        """
        :return: src, dst. each is a list of object
        - 'image': FloatTensor of shape (dt, C, H, W). resize and normalize to faster-rcnn
        - 'crop': (O, C, RH, RW) RH >= 224
        - 'bbox': (O, 4) in xyxy (0-1) / xy logw logh
        - 'trip': (T, 3)
        - 'index': (dt,)
        """
        vid_point = []
        this_index = self.index_list[index]
        vid_obj = {}
        norm_bbox = self.roidb[this_index] # (O, 2)
        bboxes = np.vstack((norm_bbox[:, 0] * self.orig_W, norm_bbox[:, 1] * self.orig_H)).T
        image, crops = self._read_image(this_index, bboxes)

        trip = self._build_graph(norm_bbox.shape[0])
        vid_obj['index'] = this_index
        vid_obj['image'] = image
        vid_obj['crop'] = crops
        vid_obj['bbox'] = torch.FloatTensor(norm_bbox)
        vid_obj['trip'] = trip
        valid = np.arange(3, dtype=np.int64)
        vid_obj['info'] = (self.orig_W, self.orig_H, valid)
        vid_obj['valid'] = torch.LongTensor(valid)

        vid_point.append(vid_obj)
        return vid_point

    def _read_image(self, index, bboxes):
        image_path = os.path.join(self.image_dir, index) + self.ext
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                dst_image = self.transform(image.convert('RGB'))
                crops = self._crop_image(index, image, bboxes)
        return dst_image, crops

    def _crop_image(self, index, image, box_center):
        crop_obj = []
        x1 = box_center[:, 0] - self.box_rad
        y1 = box_center[:, 1] - self.box_rad
        x2 = box_center[:, 0] + self.box_rad
        y2 = box_center[:, 1] + self.box_rad
        bbox = np.vstack((x1, y1, x2, y2)).transpose()
        for d in range(len(box_center)):
            crp = image.crop(bbox[d]).convert('RGB')
            crp = self.transform(crp)
            crop_obj.append(crp)
        crop_obj = torch.stack(crop_obj)
        return crop_obj

    def _build_graph(self, num_obj):
        all_trip = np.zeros([0, 3], dtype=np.float32)
        for i in range(num_obj):
            for j in range(num_obj):
                trip = [i, 0, j]
                all_trip = np.vstack((all_trip, trip))
        return torch.FloatTensor(all_trip)


def dt_collate_fn(batch):
    """
    :return: src dst. each is a list with dict element
    - 'index': list of str with length N
    - 'image': list of FloatTensor in shape (Dt, V, 1, C, H, W)
    - 'crop': list of FloatTensor in shape (Dt, V, o, C, RH, RW)
    - 'bbox': (Dt, V, o, 4)
    - 'trip': (Dt, V, t, 3)
    """
    key_set = ['index', 'image', 'crop', 'bbox', 'trip', 'valid']
    all_batch = {}
    dt = len(batch[0])
    V = len(batch)
    for key in key_set:
        all_batch[key] = []

    for f in range(dt):
        for v in range(len(batch)):
            frame = batch[v][f]
            for key in key_set:
                all_batch[key].append(frame[key])

    for key in all_batch:
        if key == 'index':
            continue
        if key in ['image', 'crop']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, 3, tensor.size(-2), tensor.size(-1))
        elif key in ['bbox', 'trip', 'valid']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, tensor.size(-1))
        else:
            print('key not exist', key)
            raise KeyError

    return all_batch


def build_vid_loaders(args):
    dset_kwargs = {
        'max_samples': None,
        'dt': 1,
        'radius': args.radius,
        'training': args.is_train,
        'mod': 'rgb' # args.modality
    }
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'collate_fn': dt_collate_fn,
    }

    common_list = 'examples/'
    dset_kwargs['image_dir'] = common_list
    dset_kwargs['training'] = args.is_train
    dset_kwargs['list_path'] = common_list + 'demo_ss.txt'
    # dset_kwargs['max_samples'] = args.num_val_samples
    # loader_kwargs['shuffle'] = args.shuffle_val
    if not os.path.exists(dset_kwargs['list_path']):
        print('not exists', dset_kwargs['list_path'])
        raise FileExistsError
    train_dset = DemoShapeStacks(**dset_kwargs)

    loader = DataLoader(train_dset, **loader_kwargs)

    return loader


if __name__ == '__main__':
    # from  import parser_helper
    from cfgs.train_cfgs import TrainOptions
    args = TrainOptions().parse()
    # from cfgs.test_cfgs import TestOptions
    # args = TestOptions().parse()
    torch.manual_seed(123)
    args.batch_size=1
    args.dt=1
    train_loader = build_vid_loaders(args)
    key_set = ['index', 'image', 'crop', 'bbox', 'trip']

    for batch in train_loader:
        for key in key_set:
            if key == 'index':
                print(batch[key])
            else:
                pass
                print('size', key, batch[key].size())
        break
