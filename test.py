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
# import _init_path
from PerceptualSimilarity.models import dist_model as dm


def quan(batch, model, evaluator, args):
    num_future = int(args.test_mod.split('_')[-1])
    num_step = args.dt
    gt_images = vis_utils.convert_batch2cv(batch['image'][1:num_step])

    best_box = None;
    box_image = None
    best_perc = None;
    perc_image = None;

    for nz in range(num_future):
        with torch.no_grad():
            inception = model.forward_inception(batch, num_step)

        images = vis_utils.convert_batch2cv(inception['pred_recon'][0:num_step - 1])

        err_list = evaluator.perceptual_metric(batch['image'][1: num_step], inception['pred_recon'][0: num_step - 1])
        if best_perc is None or np.mean(err_list) < np.mean(best_perc):
            best_perc = err_list
            perc_image = {'image': images, 'box': inception['bbox'][0: num_step - 1]}
        err_list = evaluator.calc_total_bbox_error(batch['bbox'][1: num_step], inception['bbox'][0:num_step - 1])
        if best_box is None or np.mean(err_list) < np.mean(best_box):
            best_box = err_list
            box_image = {'image': images, 'box': inception['bbox'][0:num_step - 1]}

    evaluator.push_pix_error(best_perc, 'perc', 'frame')
    evaluator.push_box_error(best_box, 'total', 'box_center')

    gt = vis_utils.convert_batch2cv(batch['image'][0:1])
    images = vis_utils.get_bbox_traj_image(box_image['box'], num_step - 1, is_torch=False, canvas=gt[0])
    evaluator.save_vid_traj(images, batch['index'][0], suff='box')

    evaluator.save_cmp_snapshot((perc_image['image'],), batch['index'][0], stride=3, suff='perc')
    evaluator.save_seq2gif(perc_image['image'], batch['index'][0], suff='perc')

    evaluator.save_raw_box_image(box_image, batch['index'][0], suff='box')
    evaluator.save_raw_box_image(perc_image, batch['index'][0], suff='perc')


def main(args):
    torch.manual_seed(123)
    np.random.seed(123)

    data_loader = model_utils.build_loaders(args)  # change to image
    model = model_utils.build_all_model(args)  # CNN, GCN, Encoder, Decoder

    # 'LPIPS'
    metric = dm.DistModel()
    metric.initialize(model='net-lin', net='alex', use_gpu=True, version='0.1')

    save_name = os.path.join(args.checkpoint.split('.')[0], '%s_%s' % (args.test_mod, args.test_split))
    evaluator = Evaluator(save_name, args, name=args.dataset, metric=metric)

    cnt = 0
    for batch in data_loader:
        print('Hallucinating... %d / %d' % (cnt, len(data_loader)), batch['index'][0])
        quan(batch, model, evaluator, args)
        cnt += 1

    evaluator.draw_save_error()
    evaluator.draw_save_pix()

if __name__ == '__main__':
    args = TestOptions().parse()
    main(args)
