#!/usr/bin/env bash

python tools/infer_simple.py \
    --dataset coco \
    --cfg configs/baselines/e2e_faster_rcnn_R-50-C4_1x.yaml \
    --load_detectron data/pretrained_model/faster_r50_coco.pkl \
    --output_dir /nfs.yoda/xiaolonw/judy_folder/transfer/detectron/ \
    --image_dir /scratch/yufeiy2/nbody/3_2348/  \

#    --image_dir data/coco/images/test2014/  \
