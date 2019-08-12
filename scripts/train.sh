#!/bin/bash

set -x

python scripts/train_vid.py --mod lstm_box --gpu 1

python scripts/train_toy.py --mod factGcnEnc --exp c32_p12_n32  --content_dim 32 --pose_dim 32 --noise_dim 32 --gpu 2
python scripts/train_toy.py --mod factGcnEnc --exp c32_p02_n32  --content_dim 32 --pose_dim 2 --noise_dim 32 --gpu 4
python scripts/train_toy.py --mod factGcnEnc --exp p32_p02_n4 --content_dim 32 --pose_dim 2 --noise_dim 4 --gpu 6


python scripts/train_toy.py --mod bypass_fact --dt 8 --exp pose2  --content_dim 510 --pose_dim 2 --gpu 3
python scripts/train_toy.py --mod bypass_fact --dt 8 --exp pose2_noise4  --content_dim 510 --pose_dim 2 --noise_dim 4 --gpu 3
python scripts/train_toy.py --mod bypass_fact --dt 8 --exp pose2_noise4_content32  --content_dim 32 --pose_dim 2 --noise_dim 4 --gpu 3

python scripts/train_toy.py --dataset v3 --mod dt_toy --dt 8 --exp v3_smooth --smooth_eps 1e-2 --gpu 1
python scripts/train_toy.py --dataset v3 --mod bypass_fact --dt 8 --exp v3_smooth --smooth_eps 1e-2 --gpu 3
python scripts/train_toy.py --dataset v3 --mod bypass_fact --dt 8 --exp v3_smooth_noise --smooth_eps 1e-2 --noise_dim 4 --gpu 5

python scripts/train_toy.py --mod bypass_fact --dt 8 --exp smoother_noise --smooth_eps 1e-2 --noise_dim 4 --gpu 1

python scripts/train_toy.py \
    --mod dt_toy  \
    --batch_size 32 \
    --dt 1 \
    --exp box \
    --gpu 1

python scripts/train_toy.py \
    --mod dt_toy  \
    --batch_size 4 \
    --dt 8 \
    --exp box \
    --gpu 0

python scripts/train_toy.py \
    --mod bypass_toy  \
    --batch_size 4 \
    --dt 8 \
    --exp box \
    --gpu 2

python scripts/train_toy.py \
    --mod bypass_toy  \
    --batch_size 4 \
    --dt 8 \
    --exp box_50012 \
    --content_dim 500 \
    --pose_dim 12 \
    --gpu 3

python scripts/train_toy.py \
    --mod bypass_toy  \
    --batch_size 4 \
    --dt 8 \
    --exp box_noise \
    --content_dim 500 \
    --pose_dim 12 \
    --noise_dim 4 \
    --gpu 1


python scripts/train_toy.py \
    --mod dt_bs  \
    --batch_size 4 \
    --dt 8 \
    --exp box \
    --gpu 2


python scripts/train_toy.py \
    --mod dt_bs  \
    --batch_size 16 \
    --dt 2 \
    --exp dt2_bs \
    --gpu 2


python scripts/train_toy.py \
    --mod dt_bs  \
    --batch_size 4 \
    --dt 8 \
    --exp dt8_bs \
    --gpu 0

python scripts/train_toy.py \
    --mod fact_toy  \
    --batch_size 4 \
    --dt 8 \
    --exp dt8 \
    --gpu 1

python scripts/train_toy.py \
    --mod bypass_toy  \
    --batch_size 4 \
    --dt 8 \
    --exp dt8 \
    --gpu 2

