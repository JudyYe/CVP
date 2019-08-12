#!/usr/bin/env bash


set -x

GPU=$1
MODEL=$2
ITER=$3
LEVEL=$4
DATASET='ss3'
# ITER=50000
# DATASET='pennOriPull'
DIR=/nfs.yoda/xiaolonw/judy_folder/transfer/pred_output

python scripts/plot_time_curve.py --gpu $GPU --dataset $DATASET \
    --dt 16  --enc_long_term 15 --test_mod best_100 --test_level $LEVEL \
    --checkpoint \
${DIR}/${MODEL}_iter${ITER}000.pth
