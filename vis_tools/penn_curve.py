# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse

import numpy as np
import os
import pickle as pkl

import matplotlib
matplotlib.use('Agg')
font = {
        'size': 16}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt


error_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output/'
max_time = 7
cite_lp = '[6]'
# compare = 'enc'
# compare = 'encBest'
# compare = 'pred'
# compare = 'dec'



def parser_helper():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str)
    parser.add_argument('-m', type=str)
    args = parser.parse_args()
    return args

def plot_shade(plot_file, ax, color=None, linestyle='-'):
    with open(plot_file, 'rb') as fp:
        error = pkl.load(fp)
    print(error.keys())
    # max_time = -1
    pref = name + '_' + obj + '_' + tstr
    # for key in error:
    #     if key.startswith(pref):
    #         try:
    #             dt = int(key.split('_%s' % tstr)[-1])
    #         except ValueError:
    #             continue
    #         if dt + 1 > max_time:
    #             max_time = dt + 1

    # save to mean, var
    mean = np.zeros(max_time)
    sigma = np.zeros(max_time)
    for t in range(max_time):
        key = pref + '%d' % t
        mean[t] = np.mean(error[key])
        sigma[t] = np.std(error[key])

    if color is not None:
        line = linestyle[0:2]
        marker = None if len(linestyle) <= 2 else linestyle[2]
        line, = ax.plot(range(1, max_time + 1), mean, color=color, linestyle=line, marker=marker)
    else:
        line, = ax.plot(range(1, max_time + 1), mean)
    alpha = 1.96
    ax.fill_between(range(1, max_time + 1), mean - alpha * sigma, mean + alpha * sigma, alpha=0.5)
    return ax, line


def multi_plot(error_list):
    error_local_list = [error_dir + e + '/test/%s.pkl' % cnt_fname for e in error_list]

    fig, ax = plt.subplots(1)
    for f, fname in enumerate(error_local_list):
        if 'Det' in fname:
            ax, line = plot_shade(fname, ax, line.get_color(), '--')
        elif 'Adv' in fname:
            ax, line = plot_shade(fname, ax, line.get_color(), '-.*')
        else:
            ax, line = plot_shade(fname, ax)
    plt.grid(True)

    plt.legend(legends, loc=2, frameon=False)
    plt.title(title, fontweight='bold')
    plt.xlabel('Time Step')

    if obj in ['ssim', 'psnr']:
        ylabel = 'Average %s' % obj.upper()
    elif obj == 'total' and name == 'box_center':
        ylabel = 'Average Location'
    elif obj == 'perc':
        ylabel = 'Average LPIPS'
    else:
        raise NotImplementedError
    plt.ylabel(ylabel)

    save_file = '/nfs.yoda/xiaolonw/judy_folder/transfer/fig/' + '%s_%s_%s.png' % (compare, name, obj)
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    plt.savefig(save_file, bbox_inches='tight')
    print(save_file)

if __name__ == '__main__':
    args = parser_helper()
    compare = args.m
    if compare == 'gym':
        error_list = [
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            # 'pennAugRectDetGymUdE_rbt_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            # 'pennAugRectGymUdE_rbtAdv1e-4_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_rbtAdv1e-4_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter300000/best_100_S0',
            'pennAugRectDetGymUdE_rbt_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter300000/best_100_S0',
            'pokAugRectGymUdE_tmp_pokFGan_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000'
            'pokAugRectGymUdE_rbtRes_pokVaePos_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0/',
            'pennAugRectGymUdE_rbt_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S3',
            'pennAugRectGymFC_fair_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt8_iter200000/best_100_S0',
            # 'pokAugRectGymUdE_rbt_pokVae_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            # 'pokAugRectGymUdE_rbt_pokVGan_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1000_11No_dt8_iter200000/best_100_S0',
        ]
        legends = ['Ours',  'Ours+Adv', 'Ours(Det)', 'Pose Knows [35]',  'LP %s' % cite_lp, 'No Factor [23]',  'No Edge']
    elif compare == 'enc':
        error_list = [
            'pennAugRectGymUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_rbt_skipDst_C0P32Z8_iid_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S0',
            'pennAugRectGymUdE_rbt_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt8_iter200000/best_100_S3',
        ]
        legends = ['Factored-200K(ours)', 'FP',  'LP %s' % cite_lp]

    for each in args.t.split(','):
        if each == 'b':
            name = 'box_center';
            cnt_fname = 'error_cnt'
            obj = 'total';
            tstr = ''
            title = r'l2 error of Predicted Location'

            multi_plot(error_list)
        elif each == 'f':
            word = 'Predicted Frames'
            name = 'frame'; cnt_fname = 'pix_cnt'

            obj = 'perc'; tstr = ''
            title = 'LPIPS of %s' % word
            multi_plot(error_list)



    # name = 'frame'; cnt_fname = 'pix_cnt'
    # obj = 'ssim'; tstr = ''
    # title = 'SSIM - t'

    # name = 'frame'; cnt_fname = 'pix_cnt'
    # obj = 'psnr'; tstr = ''
    # title = 'PSNR - t'

    # name = 'frame'; cnt_fname = 'pix_cnt'
    # obj = 'l1'; tstr = ''
    # title = 'L1 - t'

    # multi_plot(error_list)