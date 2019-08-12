#!/usr/bin/env bash

default:
    mod: skip
    dt: 16
    encoder: onlycell_image
    predictor: fact_gc
    decoder: bnRelu, 224, bnrs
    lr: 1e-4
    pose_dim: 32
    noise_dim: 8
    loss:
        l1_dst: s1
        kl: 1e-2
        bbox: 1e+2

ss3{
# ours: 1
python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --exp fukubase \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --exp fukusave \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --exp fukuback \
    --gpu 1


# encoder baseline
python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --encoder copy_image \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --encoder iid_image \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder lstm_lp \
    --exp fuku \
    --gpu 1


# decoder baseline
python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --dec_dims 256,128,64,32,16 \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --dec_dims 256,128,64,32 \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --decoder pixSig \
    --exp fuku \
    --gpu 1

# predictor:
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --mod skip_bs \
    --decoder skipNoSp \
    --graph fact_fc \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --gconv_unit_type noEdge \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --gconv_unit_type node \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --graph fact_in  \
    --exp fuku \
    --gpu 1
} ss3



--
penn {

python scripts/train_vid_zero_grad.py \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gpu 1


# no spatial
python scripts/train_vid_zero_grad.py \
    --dataset pennAugRectGym --dt 8 --bs 8 --epoches 630 --mod skip_bs \
    --modality FC \
    --decoder skipNoSp \
    --exp fuku \
    --graph fact_fc \
    --gpu 1


# No edge
python scripts/train_vid_zero_grad.py \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gconv_unit_type noEdge \
    --gpu 1

#python scripts/train_vid_zero_grad.py \
#    --dataset pennAugRectGym --modality All --dt 8 --bs 8 --epoches 630 \
#    --exp fuku \
#    --gpu 1

# Pull Up
python scripts/train_vid_zero_grad.py \
    --dataset pennOvftPull --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gpu 1

# Tennis
python scripts/train_vid_zero_grad.py \
    --dataset pennOvftTenn --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gpu 1
# Baseball
python scripts/train_vid_zero_grad.py \
    --dataset pennOvftBase --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gpu 1
# Bowl
python scripts/train_vid_zero_grad.py \
    --dataset pennOvftBowl --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gpu 1
# Golf
python scripts/train_vid_zero_grad.py \
    --dataset pennOvftGolf --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fuku \
    --gpu 1
} baseline


--

python scripts/train_vid_zero_grad.py \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --pose_dim 128 --noise_dim 32 \
    --exp fuku \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOvftPull --modality UdE --dt 8 --bs 8 --epoches 630 \
    --pose_dim 128 --noise_dim 32 \
    --exp fuku \
    --gpu 1