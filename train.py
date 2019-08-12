# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import torch
from cfgs.train_cfgs import TrainOptions
from utils import model_utils
import cvp.vis as vis_utils
from cvp.logger import Logger
from cvp.losses import LossManager


def main(args):
    torch.manual_seed(123)
    model_name = model_utils.get_model_name(args)
    float_dtype = torch.cuda.FloatTensor

    train_loader = model_utils.build_loaders(args)  # change to image

    model = model_utils.build_all_model(args)  # CNN, GCN, Encoder, Decoder
    model.type(float_dtype)
    model.train()
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    t, epoch = 0, 0
    checkpoint = {
        'args': args.__dict__,
        'counters': {
            't': None,
            'epoch': None,
        },
        'optim_state': None,
        'model_state': None, 'model_best_state': None,
        'best_t': [],
    }

    logger = Logger(model_name, args)
    loss_mng = LossManager(args)

    if args.dataset == 'ss3':
        save_iters = [50000]
    elif args.dataset.startswith('penn'):
        save_iters = [100000, 300000]
    else:
        raise NotImplementedError

    while True:
        if t >= save_iters[-1]:
            break
        for batch in train_loader:
            predictions = model(batch)  # in testing: no dst...
            total_loss_list, losses = loss_mng.separate_losses(batch, predictions)

            optimizer.zero_grad()
            total_loss_list[0].backward(retain_graph=True)
            if args.dec_zero_grad:
                model.decoder.zero_grad()
            total_loss_list[1].backward()
            optimizer.step()

            if t % args.print_every == 0:
                logger.print(t, epoch, losses, total_loss_list[0] + total_loss_list[1])
            if t % args.curve_log_every == 0:
                logger.add_loss(t, losses, pref='train/')
            if t < 1e4 and t % args.image_log_every == 0 or t % 1e4 == 0:
                images = vis_utils.get_bbox_traj(batch['bbox'][1:args.dt], predictions['bbox'][0:args.dt-1], args.dt - 1, 4)
                logger.add_images(t, images, name='train/bbox_traj')
                images = vis_utils.get_crop(predictions['real_recon'], max_num=1)
                logger.add_images(t, images, name='real_recon')
                images = vis_utils.get_crop(predictions['pred_recon'], max_num=1)
                logger.add_images(t, images, name='pred_recon')
                images = vis_utils.get_crop(batch['image'], max_num=1)
                logger.add_images(t, images, name='gt')

                images = vis_utils.get_crop(predictions['real_maskl'][:, :, -1:, :, :, :], max_num=1)
                logger.add_images(t, images, name='mask')

                if 'real_maskr' in predictions:
                    images = vis_utils.get_crop(predictions['real_maskr'], max_num=1)
                    logger.add_images(t, images, name='maskr')

            if t in save_iters:
                checkpoint['model_state'] = model.state_dict()
                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                # checkpoint['args']
                checkpoint_path = os.path.join(args.output_dir, '%s_iter%d.pth' % (model_name, t))
                print('Saving checkpoint to ', checkpoint_path)
                if not os.path.exists(os.path.dirname(checkpoint_path)):
                    os.makedirs(os.path.dirname(checkpoint_path))
                    print('## Make Directory: ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)
                print('Done saved checkpoint to ', checkpoint_path)
            t += 1

        epoch += 1

    checkpoint['model_state'] = model.state_dict()
    checkpoint['optim_state'] = optimizer.state_dict()
    checkpoint['counters']['t'] = t
    checkpoint['counters']['epoch'] = epoch
    checkpoint_path = os.path.join(args.output_dir, '%s_iter%d.pth' % (model_name, t))
    print('Saving checkpoint to ', checkpoint_path)
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
        print('## Make Directory: ', checkpoint_path)
    torch.save(checkpoint, checkpoint_path)
    print('Done saved checkpoint to ', checkpoint_path)


if __name__ == '__main__':
    args = TrainOptions().parse()
    main(args)