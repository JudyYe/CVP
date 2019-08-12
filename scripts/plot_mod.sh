#!/usr/bin/env bash


set -x

GPU=$1
MODEL=$2
ITER=$3
DATASET=$4
MOD=$5
LEVEL=$6
# ITER=50000
# DATASET='pennOriPull'
DIR=/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output

python scripts/plot_time_curve.py --gpu $GPU --dataset $DATASET \
    --dt 16  --enc_long_term 15 --test_mod $MOD --test_level $LEVEL \
    --checkpoint \
${DIR}/${MODEL}_iter${ITER}000.pth

#${DIR}/ss3rgb_museum_skip_C0P32norm_iid_res_reluOne765bnrs_fact_gc_n2e2n_Ppix1_11_dt16_iter${ITER}.pth

#${DIR}/${DATASET}rgb_museum_skip_C0P32norm_copy_res_reluOne765bnrs_fact_gc_n2e2n_Ppix1_11_dt16_iter${ITER}.pth\


#python scripts/plot_traj.py --gpu $GPU --dataset $DATASET \
#    --dt 16  --enc_long_term -1 --test_mod all \
#    --checkpoint \
#${DIR}/${DATASET}rgb_bn_skip_C0P128norm_onlycell_image_sigOne765bnrs_fact_gc_n2e2n_Ppix1_11_dt16_iter${ITER}.pth \


#
### This is for penn
#GPU=$1
#ITER=100000
#DATASET='pennOriGym'
## ITER=50000
## DATASET='pennOriPull'
#DIR=/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output
#
#python scripts/test_vid_skeleton.py --gpu $GPU --dataset $DATASET \
#    --dt 8  --enc_long_term -1 --test_mod ae\
#    --checkpoint \
#${DIR}/${DATASET}UdE_splatAE_skip_C0P32norm_onlycell_res_maskSkipSplat_fact_gc_n2e2n_11_dt8_iter${ITER}.pth\
