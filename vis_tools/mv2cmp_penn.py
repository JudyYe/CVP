# --------------------------------------------------------
# Graph as Label
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
dst_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/save_cmp/'
gt_dir = src_dir + 'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/gt_S0/'

def parser_helper():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='appr')
    parser.add_argument('-l', type=str, default='S0')
    parser.add_argument('-f', type=str, default='none')
    parser.add_argument('-m', type=str)
    parser.add_argument('-s', type=str, default='test')
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--mv', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--gt', action='store_true')
    args = parser.parse_args()
    return args

def cmp_gif(fname_list):
    for index in fname_list:
        # fname = env_ccs-hard-h=3-vcom=2-vpsf=0-v=244_snap.gif
        gif_list = []
        name = ''
        src_file = os.path.join(gt_dir, args.s, index + '_init.jpg')
        print(src_file)
        init_img = cv2.imread(src_file)
        init_img = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        init_img = put_text(init_img, 'Initial Frame')

        dst_file = os.path.join(dst_dir, index, 'init.jpg')
        if not os.path.exists(os.path.dirname(dst_file)):
            os.makedirs(os.path.dirname(dst_file))
            print('## Make Direcotry', dst_file)

        # cv2.imwrite(dst_file, init_img)
        # exit()
        # cmd = 'cp %s %s ' % (src_file, dst_file)
        # os.system(cmd)

        print_legend = []
        if args.gt:
            gt_file = os.path.join(gt_dir, args.s, index + '_snappix.gif')
            gif = imageio.mimread(gt_file)
            gif_list.append(gif)
            name += '_GT'
            print_legend.append('GT')
        for i, model in enumerate(error_list):
            # src_list = glob.glob(os.path.join(src_dir, model, '%s_%s' % (args.t, args.l), args.s, index + '*.gif'))
            # print(os.path.join(src_dir, model,  args.s, index + '*.gif'))
            src_list = glob.glob(os.path.join(src_dir, model,  args.s, index + '_*.gif'))
            # print(src_list)
            if len(src_list) == 0:
                print('no model', os.path.join(src_dir, model, args.s, index + '_*.gif'))
                continue
            assert len(src_list) == 1
            gif = imageio.mimread(src_list[0])
            gif_list.append(gif)
            name += '_' + legends[i]
        print_legend.extend(legends)
        if args.gt:
            gif_list = sort_merge_gif(gif_list, print_legend, init_img)
        else:
            gif_list = sort_merge_gif(gif_list, print_legend)
        if gif_list is None:
            continue
        dst_file = os.path.join(dst_dir, index, compare + name + '.gif')
        gif_list = [each.astype(np.uint8) for each in gif_list]
        imageio.mimsave(dst_file, gif_list, subrectangles=True, duration = 1/8)

        src_file = os.path.join(gt_dir, args.s, index + '_init.jpg')
        dst_file = os.path.join(dst_dir, index, 'init.jpg')
        cmd = 'cp %s %s ' % (src_file, dst_file)
        os.system(cmd)

def sort_merge_gif(gif_list, legend, init=None):
    N = len(gif_list)
    if N != len(legend):
        return
    # assert N == len(legend), '%d %d' % (N, len(legend))
    T = len(gif_list[0])
    big_list = []
    for t in range(T):
        merge = init
        for n in range(N):
            image = put_text(gif_list[n][t][:, :, 0:3], legend[n])
            merge = hstack((merge, image))
        big_list.append(merge)
    return big_list

def put_text(rgb, text, m=50):
    h, w = rgb.shape[0: 2]
    plain = np.ones([m, w, 3]) * 255
    cv2.putText(plain, text, (10, m-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    rgb = vstack((plain, rgb))
    return rgb


def link_to(fname_list):
    for index in fname_list:
        if args.gt:
            gt_file = os.path.join(gt_dir, args.s, index + '_box.jpg')
            dst_file = os.path.join(dst_dir, index, 'gt_box.jpg')
            cmd = 'cp %s %s' % (gt_file, dst_file)
            os.system(cmd)
        # fname = env_ccs-hard-h=3-vcom=2-vpsf=0-v=244_snap.gif
        for model in error_list:
            src_list = glob.glob(os.path.join(src_dir, model, args.s, index + '_*'))
            # print(src_list)
            # src_file = os.path.join(src_dir, model, args.s, fname)
            dst_group = os.path.join(dst_dir, index + '/')
            model = os.path.dirname(model)
            rtn = cmd_link_list(src_list, dst_group, index, model)

            if rtn == 0:
                print(os.path.join(src_dir, model, args.s, index + '*'))


def cmd_link_list(src_list, dst_dir, index, model):
    if len(src_list) == 0:
        print('Not Ready!')
        return False
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print('## Make Directory', dst_dir)

    for full_name in src_list:
        suff = full_name.split(index + '_')[-1]
        ext = '.' + suff.split('.')[-1]
        pref = suff.split('.')[0]
        # print(pref)
        dst = os.path.join(dst_dir, pref + '_' + model + ext)
        # print('dst_dir', dst_dir)
        # print('pref', pref, model)
        # print('dst', dst)
        if os.path.exists(dst):
            cmd = 'rm %s' % dst
            os.system(cmd)

        # full_name = full_name.replace(src_dir, '../../pred_output/')
        cmd = 'cp %s %s ' % (full_name, dst)
        # print(cmd)
        os.system(cmd)
    return True


def cmd_link(src, dst):
    if not os.path.exists(src):
        print('Not Ready: %s' % src)
        return
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
        print('## Make Directory: ', dst)
    if os.path.exists(dst):
        cmd = 'rm %s' % dst
        os.system(cmd)

    cmd = 'ln -s %s %s ' % (src, dst)
    os.system(cmd)




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
        new_img_list.append(np.vstack((img, zeros)))
    merge = img_list[0]
    for i in range(1, len(img_list)):
        img = new_img_list[i]
        zeros = np.ones([h, m, 3]) * 255
        merge = np.hstack((merge, zeros, img))
    return merge


if __name__ == '__main__':
    args = parser_helper()

    compare = args.m
    if compare == 'gym':
        error_list = [
            'pokAugRectGymUdE_tmp_pokFGan_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000'
            'pokAugRectGymUdE_rbtRes_pokVaePos_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0/',
            # 'pok64AugRectGymUdE_rbt_pokVGan_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_rbt_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S3',
            'pennAugRectGymFC_fair_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_rbtAdv1e-4_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectDetGymUdE_rbt_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter300000/best_100_S0',
        ]
        legends = ['Pose Knows', 'LP', 'No Factor', 'No Edge', 'Ours',  'Ours(+Adv)', 'Ours(Det)']

    if args.f == 'none':
        demo_file = '/nfs.yoda/xiaolonw/judy_folder/transfer/demo_list.txt'
        with open(demo_file) as fp:
            fname = [line.strip() for line in fp]
    else:

        fname = args.f.split(',')
    if args.mv:
        link_to(fname)
    if args.gif:
        cmp_gif(fname)