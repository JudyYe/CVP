# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import os

import numpy as np
import torch
from cfgs.test_cfgs import TestOptions
from utils import model_utils
import cvp.vis as vis_utils
from cvp.evaluator import Evaluator


def multi(batch, model, evaluator, args):
    num_step = args.dt
    num_future = 5
    multi_box = []
    multi_image = []
    for nz in range(num_future):
        with torch.no_grad():
            inception = model.forward_inception(batch, num_step, seed=None)

        gt = vis_utils.convert_batch2cv(batch['image'][0:1])
        images = vis_utils.get_bbox_traj_image(inception['bbox'][0:num_step - 1], num_step - 1,
                                         is_torch=False, canvas=gt[0])
        multi_box.extend(images)

        images = vis_utils.convert_batch2cv(inception['pred_recon'][0: num_step - 1])
        multi_image.append(images)

    evaluator.save_pack(multi_image, multi_box, batch['index'][0])


def main(args):
    torch.manual_seed(123)
    np.random.seed(123)

    data_loader = model_utils.build_loaders(args)  # change to image
    model = model_utils.build_all_model(args)  # CNN, GCN, Encoder, Decoder

    save_name = os.path.join(args.checkpoint.split('.')[0], '%s_%s' % (args.test_mod, args.test_split))
    evaluator = Evaluator(save_name, args, name=args.dataset)

    for batch in data_loader:
        print('Hallucinating...', batch['index'][0])
        multi(batch, model, evaluator, args)


if __name__ == '__main__':
    args = TestOptions().parse()
    args.dataset = 'demo'
    main(args)
