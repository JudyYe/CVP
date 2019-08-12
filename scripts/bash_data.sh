#!/usr/bin/env bash

set -x
#ITER=30000
#DATASET=ss3
#DIR=/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output
#python scripts/test_vid.py --gpu 0 --dataset $DATASET \
#    --checkpoint \
#$DIR/ss3rgb_AbRand_bg_C128P32norm_copy_image_shareBg_fact_gc_n2e2n_Ppix10_11_dt16_iter${ITER}.pth \


#$DIR/ss3rgb_tmp_multiGraph_C128P32norm_onlycell_box_shareBg_fact_gc_mulEdge_11_dt16_iter${ITER}.pth \



#ITER=30000
#DATASET=ss3
#DIR=/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output


#nice -19 python scripts/test_vid_with_cnn.py --gpu 0 --dataset $DATASET \
#    --checkpoint \
#$DIR/ss3rgb_multiGraph_bg_C128P32norm_onlycell_image_shareBg_fact_gc_mulEdge_Pfea1 00_11_dt16_iter${ITER}.pth \



## This is for penn
GPU=$1
MODEL=$2
ITER=$3
#SPLIT=$4
MOD=$4
DATA=$5
DATASET='pennAug'$DATA
# ITER=50000
# DATASET='pennOriPull'
DIR=/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output

python scripts/test_vid_skeleton.py --gpu $GPU --dataset $DATASET \
    --dt 8  --enc_long_term -1 --test_mod $MOD --test_split test \
    --checkpoint \
${DIR}/${MODEL}_iter${ITER}000.pth
