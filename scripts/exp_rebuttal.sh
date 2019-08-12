#!/usr/bin/env bash

python scripts/train_vid_adv.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbtAdv \
    --gpu 1

# det
python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectDetGym --modality UdE --dt 8 --bs 8 --epoches 730 \
    --exp rbtBack \
    --gpu 1

# reproduce Jacob
python scripts/train_pok.py  \
    --dataset pokAugRectGym --mod pokVae --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbt \
    --gpu 1

python scripts/train_pok.py  \
    --dataset pokAugRectGym --mod pokVaeVel --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbt \
    --gpu 1


python scripts/train_pok.py  \
    --dataset pokAugRectGym --mod pokVaePos --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbt \
    --gpu 1

python scripts/train_pok_gan.py  \
    --dataset pok64AugRectGym --mod pokVGan --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbt \
    --gpu 1


python scripts/train_pok_gan.py  \
    --dataset pokAugRectGym --mod pokVGan --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbt \
    --gpu 1
# LP
python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp rbt \
    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder iid_image \
    --exp rbt \
    --gpu 1


python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder lstm_lp \
    --exp rbt \
    --gpu 1