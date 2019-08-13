# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import glob

import imageio
import numpy as np
import os
import cv2


src_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output/'
dst_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/expand_cmp/'

def parser_helper():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='none')
    parser.add_argument('-m', type=str)
    parser.add_argument('-s', type=str, default='test')
    parser.add_argument('--cmp', action='store_true')
    parser.add_argument('--gt', action='store_true')
    args = parser.parse_args()
    return args


def cmp_gif(fname_list):
    for index in fname_list:
        # fname = env_ccs-hard-h=3-vcom=2-vpsf=0-v=244_snap.gif
        gif_list = []
        name = ''
        src_file = os.path.join(gt_dir, args.s, index + '_init.jpg')
        # src_file = os.path.join(gt_dir, index + '_init.jpg')
        init_img = cv2.imread(src_file)
        if init_img is None:
            print(src_file)
        init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        # init_img = put_text(init_img, 'Initial Frame')

        print_legend = []
        if args.gt:
            gt_file = os.path.join(gt_dir, args.s, index + '_snappix.gif')
            # gt_file = os.path.join(gt_dir, index + '_snap.gif')
            gif = imageio.mimread(gt_file)
            gif_list.append(gif)
            name += '_GT'
            print_legend.append('GT')
        for i, model in enumerate(error_list):
            # src_list = glob.glob(os.path.join(src_dir, model, '%s_%s' % (args.t, args.l), args.s, index + '*.gif'))
            # print(os.path.join(src_dir, model,  args.s, index + '*.gif'))
            src_list = glob.glob(os.path.join(src_dir, model,  args.s, index + '_perc.gif'))
            # print(src_list)
            if len(src_list) == 0:
                print('no model', os.path.join(src_dir, model, args.s, index + '_perc.gif'))
                continue
            assert len(src_list) == 1, os.path.join(src_dir, model, args.s, index + '_perc.gif')
            gif = imageio.mimread(src_list[0])
            gif_list.append(gif)
            name += '_' + legends[i]
        print_legend.extend(legends)
        if args.gt:
            gif_list = sort_merge_gif(gif_list, print_legend, init_img, time_ori=time_ori)
        else:
            gif_list = sort_merge_gif(gif_list, print_legend, time_ori=time_ori)
        if gif_list is None:
            continue
        gif_list = cv2.cvtColor(gif_list.astype(np.uint8), cv2.COLOR_RGB2BGR)

        dst_file = os.path.join(dst_dir, index + compare + name + '.png')
        if not os.path.exists(os.path.dirname(dst_file)):
            os.makedirs(os.path.dirname(dst_file))
            print('## Make Directory', dst_file)
        # gif_list = [each.astype(np.uint8) for each in gif_list]
        cv2.imwrite(dst_file, gif_list)
        # imageio.mimsave(dst_file, gif_list, subrectangles=True, duration = 1/8)


def sort_merge_gif(gif_list, legend, init=None, time_ori='v'):
    N = len(gif_list)
    if N != len(legend):
        return
    # assert N == len(legend), '%d %d' % (N, len(legend))
    if time_ori == 'v':
        time_stack = vstack
        method_stack = hstack
    else:
        time_stack = hstack
        method_stack = vstack
    big_list = None
    for t in strip_list:
        merge = None
        # merge = init
        # image = None
        for n in range(N):
            # image = put_text(gif_list[n][t][:, :, 0:3], legend[n])
            image = gif_list[n][t][:, :, 0:3]
            merge = method_stack((merge, image))
        big_list = time_stack((big_list, merge))
    return big_list

def put_text(rgb, text, m=50):
    h, w = rgb.shape[0: 2]
    plain = np.ones([m, w, 3]) * 255
    cv2.putText(plain, text, (10, m-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    rgb = vstack((plain, rgb))
    return rgb

def vstack(img_list, m=10):
    width = -1
    if img_list[0] is None:
        return img_list[1]

    for img in img_list:
        if width < img.shape[1]:
            width = img.shape[1]
    new_img_list = []
    for img in img_list:
        zeros = np.ones([img.shape[0], width - img.shape[1], 3]) * 255
        new_img_list.append(np.hstack((img, zeros)))
    merge = new_img_list[0]
    for i in range(1, len(new_img_list)):
        img = new_img_list[i]
        zeros = np.ones([m, width, 3]) * 255
        merge = np.vstack((merge, zeros, img))
    return merge


def hstack(img_list, m=10):
    if img_list[0] is None:
        return img_list[1]
    h = img_list[0].shape[0]
    # find max h
    for img in img_list:
        if h < img.shape[0]:
            h = img.shape[0]

    new_img_list = []
    for img in img_list:
        zeros = np.ones([h - img.shape[0], img.shape[1], 3]) * 255
        try:
            new_img_list.append(np.vstack((img, zeros)))
        except ValueError:
            print(img.shape, zeros.shape)
            exit()
    merge = img_list[0]
    for i in range(1, len(img_list)):
        img = new_img_list[i]
        zeros = np.ones([h, m, 3]) * 255
        merge = np.hstack((merge, zeros, img))
    return merge


if __name__ == '__main__':
    args = parser_helper()
    time_ori = 'v'
    compare = args.m
    if compare == 'gym':
        strip_list = [0, 3, 5, -1]
        gt_dir = src_dir + 'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/gt_S0/'
        demo_file = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo_list.txt'

        error_list = [
            'pokAugRectGymUdE_tmp_pokFGan_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000'
            'pokAugRectGymUdE_rbtRes_pokVaePos_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0/',
            'pennAugRectGymUdE_rbt_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S3',
            'pennAugRectGymFC_fair_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_rbtAdv1e-4_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter300000/best_100_S0',
            'pennAugRectDetGymUdE_rbt_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter300000/best_100_S0',
        ]
        legends = ['Pose Knows', 'LP', 'No Factor', 'No Edge', 'Ours', 'Ours+Adv', 'Ours(Det)']
    elif compare == 'num':
        strip_list = [7, -1]
        gt_dir = '/scratch/yufeiy2/shapestacks/ss_gt_vis/'
        demo_file = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo_ss.txt'

        error_list = [
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0',
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0',
        ]
        legends = ['No Edge', 'Ours']

    elif compare == 'pred':
        time_ori = 'h'
        strip_list = list(range(2, 16, 3))
        gt_dir = '/scratch/yufeiy2/shapestacks/ss_gt_vis/'
        # demo_file = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo_ss.txt'

        error_list = [
            'ss3rgb_fairPredBug_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0',
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0',
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0',
        ]
        legends = ['No Factor', 'No Edge', 'Factored(ours)']

    if args.f == 'file':
        with open(demo_file) as fp:
            fname = [line.strip() for line in fp]
    elif args.f == 'none':
        fname = [
            'env_blocks-easy-h=3-vcom=1-vpsf=0-v=6', # for enc
            'env_blocks-easy-h=3-vcom=1-vpsf=0-v=27',
            'env_ccs-hard-h=3-vcom=2-vpsf=0-v=244',
            'env_blocks-hard-h=3-vcom=1-vpsf=0-v=28',
            'env_ccs-hard-h=3-vcom=1-vpsf=0-v=285',
            'env_ccs-hard-h=3-vcom=1-vpsf=0-v=280',
            'env_ccs-hard-h=3-vcom=1-vpsf=0-v=265',
            'env_ccs-hard-h=3-vcom=2-vpsf=0-v=168',
            'env_ccs-hard-h=3-vcom=2-vpsf=0-v=292',
        ]
    else:
        fname = args.f.split(',')
    if args.cmp:
        cmp_gif(fname)