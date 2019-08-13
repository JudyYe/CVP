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
max_time = 15
cite_lp = '[6]'

# compare = 'enc'
# compare = 'encBest'
# compare = 'pred'
# compare = 'dec'


line_type = ['-', '--', ':', '-.']
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

def parser_helper():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str)
    parser.add_argument('-m', type=str)
    args = parser.parse_args()
    return args

def plot_shade(plot_file, ax, num):
    with open(plot_file, 'rb') as fp:
        error = pkl.load(fp)
    print(plot_file, error.keys())
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

    # ax.plot(range(1, max_time + 1), mean, color=color[divmod(num, period)[0]], linestyle = line_type[divmod(num, period)[1]])
    ax.plot(range(1, max_time + 1), mean)
    alpha = 1.96
    ax.fill_between(range(1, max_time + 1), mean - alpha * sigma, mean + alpha * sigma, alpha=0.5)
    return ax


def multi_plot():
    error_local_list = [error_dir + e + '/%s.pkl' % cnt_fname for e in error_list]

    fig, ax = plt.subplots(1)
    for n, fname in enumerate(error_local_list):
        ax = plot_shade(fname, ax, n)
    ax.set_xticks(range(0, max_time + 1, 3))
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
    period = 100

    if compare == 'enc':
        error_list = [
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_copy_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S3/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_iid_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
        ]
        legends = ['Ours(2)', 'No-Z(2)', 'LP-PRED(1)', 'LP-GT(16)', 'IID-GT(16)']

    elif compare == 'encBest':
        error_list = [
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_copy_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_iid_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test/',
            'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S3/test/',
        ]
        legends = ['Ours', 'No-Z', 'FP', 'LP %s' % cite_lp]

    elif compare == 'pred':
        # error_list = [
        #     'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
        #     # 'ss3rgb_fairPredBug_skipDst_bs_C0P32Z8_onlycell_image_skipNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
        #     'ss3rgb_fairPredBug_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
        #     'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/all_S0/test/',
        #     # 'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_node_Ppix1_11No_dt16_iter50000/all_S0'
        # ]
        error_list = [
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test/',
            'ss3rgb_fairPredBug_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss3/',
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss3/',
        ]
        legends = ['Factored(ours)', 'No-Factor [23]',  'No-Edge']

    elif compare == 'dec':
        error_list = [
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
            'ss3rgb_fairDec_skipDst_C0P32Z8_onlycell_image_fastSig76516batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
            'ss3rgb_fairDec_skipDst_C0P32Z8_onlycell_image_fastSig76532batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
            'ss3rgb_fairDec_skipDst_C0P32Z8_onlycell_image_pixSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0'
        ]
        legends = ['Feat-Late', 'Feat-Mid', 'Feat-Early', 'Pixel']
    elif compare == 'num':
        error_list = [
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test/',
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss4/',
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss5/',
            'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss6/',
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss3/',
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss4/',
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss5/',
            'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/best_100_S0/test_ss6/',

            # 'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S3/test/',
            # 'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S3/test_ss4/',
            # 'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S3/test_ss5/',
            # 'ss3rgb_fairEnc_skipDst_C0P32Z8_lstm_lp_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/best_100_S3/test_ss6/',

        ]
        # error_list = [
        #     'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test/',
        #     'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test_ss4/',
        #     'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test_ss5/',
        #     'ss3rgb_fairback_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_n2e2n_Ppix1_11No_dt16_iter50000/all_S0/test_ss6/',
        #     'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/all_S0/test/',
        #     'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/all_S0/test_ss4/',
        #     'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/all_S0/test_ss5/',
        #     'ss3rgb_fairPred_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt16_iter50000/all_S0/test_ss6/',
        # ]
        legends = ['Ours(3)', 'Ours(4)', 'Ours(5)', 'Ours(6)',
                   'No-Edge(3)', 'No-Edge(4)', 'No-Edge(5)', 'No-Edge(6)',
                   ]
        period = 4


    for each in args.t.split(','):
        if each == 'b':
            name = 'box_center';
            cnt_fname = 'error_cnt'
            obj = 'total';
            tstr = ''
            title = r'l2 error of Predicted Location'

            multi_plot()
        elif each == 'r':
            word = 'Reconstructed Frames'
            name = 'recon'; cnt_fname = 'pix_cnt'
            #
            # obj = 'ssim'; tstr = ''
            # title = 'SSIM of %s' % word
            # multi_plot()
            #
            # obj = 'psnr'; tstr = ''
            # title = 'PSNR of %s' % word
            # multi_plot()

            obj = 'perc'; tstr = ''
            title = 'LPIPS of %s' % word
            multi_plot()
        elif each == 'f':
            word = 'Predicted Frames'
            name = 'frame'; cnt_fname = 'pix_cnt'

            # obj = 'ssim'; tstr = ''
            # title = 'SSIM of %s' % word
            # multi_plot()
            #
            # obj = 'psnr'; tstr = ''
            # title = 'PSNR of %s' % word
            # multi_plot()

            obj = 'perc'; tstr = ''
            title = 'LPIPS of %s' % word
            multi_plot()



    # name = 'frame'; cnt_fname = 'pix_cnt'
    # obj = 'ssim'; tstr = ''
    # title = 'SSIM - t'

    # name = 'frame'; cnt_fname = 'pix_cnt'
    # obj = 'psnr'; tstr = ''
    # title = 'PSNR - t'

    # name = 'frame'; cnt_fname = 'pix_cnt'
    # obj = 'l1'; tstr = ''
    # title = 'L1 - t'

    # multi_plot()