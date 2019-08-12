#!/usr/bin/env bash

set -x

python scripts/train_traj.py \
    --exp sto \
    --dataset ss3 --epoches 100 \
    --mod traj \
    --encoder onlycell_box \
    --gpu 0

python scripts/train_traj.py \
    --exp sto \
    --dataset ss3 --epoches 100 \
    --mod traj \
    --gconv_unit_type mulEdge \
    --encoder onlycell_box \
    --gpu 0


---
python scripts/train_traj.py \
    --dataset 3body10000_10000_10 --bs 512 --epoches 100 \
    --exp graph \
    --mod traj \
    --graph fact_in \
    --encoder onlycell_box \
    --gpu 0

python scripts/train_traj.py \
    --dataset 3body10000_10000_10 --bs 512 --epoches 100 \
    --exp graph \
    --mod traj \
    --encoder onlycell_box \
    --gpu 2

python scripts/train_traj.py \
    --dataset 3body10000_10000_10 --bs 512 --epoches 100 \
    --exp graph \
    --mod traj \
    --gconv_unit_type nein \
    --encoder onlycell_box \
    --gpu 1

python scripts/train_traj.py \
    --dataset 3body10000_10_1 --bs 512 --epoches 100 \
    --exp graph \
    --mod traj \
    --encoder onlycell_box \
    --gpu 0

python scripts/train_traj.py \
    --dataset 3body10000_10_1 --bs 512 --epoches 100 \
    --exp graph \
    --mod traj \
    --graph fact_in \
    --encoder onlycell_box \
    --gpu 1

python scripts/train_traj.py \
    --dataset 3body10000_10000_1 --bs 512 --epoches 100 \
    --exp graph \
    --mod traj \
    --gconv_unit_type nein \
    --encoder onlycell_box \
    --gpu 2

---

python scripts/train_vid_zero_grad.py \
    --exp align \
    --decoder thin \
    --dec_dims 512,256,128,64 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp align \
    --decoder RegBg \
    --gpu 1


---

python scripts/train_vid_zero_grad.py \
    --exp AbRand \
    --encoder copy_image \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gpu 1

# encoder ablation
python scripts/train_vid_zero_grad.py \
    --exp AbRand \
    --encoder short_image \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gpu 1

---
# Edge Graph


python scripts/train_vid_zero_grad.py \
    --exp AbGraph \
    --graph edge_gc \
    --gpu 1


---
# mutli graph
python scripts/train_vid_zero_grad.py \
    --exp multiGraph \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp multiGraph \
    --gconv_unit_type mulEdge \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1


---
# test norm p, pixel
python scripts/train_vid_zero_grad.py \
    --exp normp \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp normp \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad False \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp normp \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad False \
    --recon_stop_grad False \
    --gpu 1

-

python scripts/train_vid_zero_grad.py \
    --exp normp \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gconv_stop_grad False \
    --recon_stop_grad False \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp normp \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gconv_stop_grad False \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp normp \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1


---
#test where to detach

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad False \
    --recon_stop_grad False \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad False \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1

-

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_pix_loss True \
    --l1_dst_loss_weight 100 \
    --gconv_stop_grad False \
    --recon_stop_grad False \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_pix_loss True \
    --l1_dst_loss_weight 100 \
    --gconv_stop_grad False \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_pix_loss True \
    --l1_dst_loss_weight 100 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1

--
python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gconv_stop_grad False \
    --recon_stop_grad False \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gconv_stop_grad False \
    --recon_stop_grad True \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp detach \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gconv_stop_grad True \
    --recon_stop_grad True \
    --gpu 1

---
python scripts/train_vid_zero_grad.py \
    --mod bg \
    --dataset ss3 \
    --exp zero \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --feat_constraint none \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --mod bg \
    --dataset ss3 \
    --exp zero \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --feat_constraint none \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --mod bg \
    --dataset ss3 \
    --exp zero \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --feat_constraint none \
    --gpu 1

-

python scripts/train_vid_zero_grad.py \
    --mod bg \
    --dataset ss3 \
    --exp zero \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --feat_constraint norm \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --mod bg \
    --dataset ss3 \
    --exp zero \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --feat_constraint norm \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --mod bg \
    --dataset ss3 \
    --exp zero \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --feat_constraint norm \
    --gpu 1

---
# test mask only depend on pose
python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp mask \
    --decoder poseMask \
    --pose_fea_loss True \
    --pose_loss_weight 10 \
    --gpu 1

---
# test pose net
python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpace \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 1 \
    --feat_constraint none \
    --gpu 1

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpace \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 10 \
    --feat_constraint none \
    --gpu 1

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpaceNoBN \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 10 \
    --feat_constraint none \
    --normalization none \
    --gpu 1

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpace \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 100 \
    --feat_constraint none \
    --gpu 1


python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpaceC1 \
    --decoder shareBg \
    --pose_fea_loss True \
    --content_loss_weight 1 \
    --pose_loss_weight 1000 \
    --feat_constraint sigmoid \
    --gpu 1


python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpace \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 10 \
    --feat_constraint norm \
    --gpu 1

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp PoseSpace \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 10 \
    --feat_constraint all_norm \
    --gpu 1

---

# 2stage training
python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp 2stage-bs \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --gpu 3

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp 2stage-bs \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 1 \
    --gpu 1

python scripts/two_stage_train_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp 2stage \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --gpu 3

python scripts/two_stage_train_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp 2stage \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 1 \
    --gpu 1

---
# test Ae of flat
python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp Exp2C1 \
    --decoder RegBg \
    --pose_pix_loss True \
    --kl_loss_weight 0 \
    --content_loss_weight 1 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0. \
    --gpu 0

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp Exp2C1NoBN \
    --decoder shareBg \
    --dec_norm none \
    --pose_pix_loss True \
    --kl_loss_weight 0 \
    --content_loss_weight 1 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0. \
    --gpu 0

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp Exp2C1 \
    --decoder shareBg \
    --pose_fea_loss True \
    --kl_loss_weight 0 \
    --content_loss_weight 1 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0. \
    --gpu 0

python scripts/train_vid_with_cnn.py \
    --mod flat \
    --dataset ss3 \
    --exp Exp2C10 \
    --decoder shareBg \
    --pose_pix_loss True \
    --kl_loss_weight 0 \
    --content_loss_weight 10 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0. \
    --gpu 1

python scripts/train_vid_with_cnn.py \
    --mod flat \
    --dataset ss3 \
    --exp flat \
    --decoder RegBg \
    --pose_fea_loss True \
    --kl_loss_weight 0 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0. \
    --gpu 2

python scripts/train_vid_with_cnn.py \
    --mod flat \
    --dataset ss3 \
    --exp flat \
    --decoder RegBg \
    --pose_fea_loss True \
    --pose_dim 128 \
    --kl_loss_weight 0 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0. \
    --gpu 2
---
# test content and pose spatial should be 1 or 2?
python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp pose \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --gpu 1

python scripts/train_vid_with_cnn.py \
    --mod flat \
    --dataset ss3 \
    --exp pose \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --gpu 3


python scripts/train_vid_with_cnn.py \
    --mod flat \
    --dataset ss3 \
    --exp pose \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 1 \
    --gpu 0

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ss3 \
    --exp pose \
    --decoder shareBg \
    --pose_fea_loss True \
    --pose_loss_weight 1 \
    --gpu 1

# test background

python scripts/train_vid_with_cnn.py \
    --mod bg \
    --dataset ssCam \
    --modality rgb \
    --exp bgToy \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 1 \
    --gpu 3



python scripts/train_vid_with_cnn.py \
    --mod bg \
    --modality rgb \
    --dataset ssCam \
    --exp bgToy1e-1 \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight .1 \
    --gpu 1


python scripts/train_vid_with_cnn.py \
    --mod bg \
    --modality rgb \
    --dataset ssCam \
    --exp bgToy1e10 \
    --decoder shareBg \
    --pose_pix_loss True \
    --l1_dst_loss_weight 10 \
    --gpu 2


---

# this group of exp is to test if we should constraint pose in pixel level or feature level. In pixel level, we need bbox working as well, so ...

python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp DetachFea1e-0 \
    --encoder onlycell_image \
    --content_loss_weight 0.1 \
    --pose_loss_weight 1 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 0 \
    --gconv_pixel_loss False \
    --gpu 0

python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp DetachFea1e-1 \
    --encoder onlycell_image \
    --content_loss_weight 0.1 \
    --pose_loss_weight 0.1 \
    --l1_src_loss_weight 1 \
    --l1_dst_loss_weight 0 \
    --gconv_pixel_loss False \
    --gpu 3

python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp DetachFea1e-2 \
    --encoder onlycell_image \
    --content_loss_weight 0.1 \
    --pose_loss_weight 0.01 \
    --l1_src_loss_weight 1 \
    --l1_dst_loss_weight 0 \
    --gconv_pixel_loss False \
    --gpu 3


python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp DetachPix1e-0 \
    --encoder onlycell_image \
    --content_loss_weight 0.1 \
    --pose_loss_weight 0.1 \
    --l1_src_loss_weight 1. \
    --l1_dst_loss_weight 1. \
    --gconv_pixel_loss True \
    --gpu 3


python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp DetachPix1e-1 \
    --encoder onlycell_image \
    --content_loss_weight 0.1 \
    --pose_loss_weight 0.1 \
    --l1_src_loss_weight 1 \
    --l1_dst_loss_weight 0.1 \
    --gconv_pixel_loss True \
    --gpu 3


#
#python scripts/train_vid_with_cnn.py \
#    --mod lstm_layout \
#    --decoder shareContent \
#    --obj_size 224,224 \
#    --exp DetachPix1e-2 \
#    --encoder onlycell_image \
#    --content_loss_weight 0.1 \
#    --pose_loss_weight 0 \
#    --l1_src_loss_weight 1 \
#    --l1_dst_loss_weight 0.1 \
#    --gconv_pixel_loss True \
#    --gpu 3
---

#this group of exp is to test if the detach helps when we add back pose
python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp AeP1e-1Detach \
    --encoder onlycell_image \
    --kl_loss_weight 0 \
    --content_loss_weight 0 \
    --pose_loss_weight 0.1 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --gpu 3


python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp AeP1e-2Detach \
    --encoder onlycell_image \
    --kl_loss_weight 0 \
    --content_loss_weight 0 \
    --pose_loss_weight 0.01 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --gpu 3

python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp AeP1e-0Detach \
    --encoder onlycell_image \
    --kl_loss_weight 0 \
    --content_loss_weight 0 \
    --pose_loss_weight 1 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --gpu 0


---
python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp AeC1e-1 \
    --encoder onlycell_image \
    --kl_loss_weight 0 \
    --content_loss_weight 0.1 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --gpu 0



python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp AeC1e-0 \
    --encoder onlycell_image \
    --kl_loss_weight 0 \
    --content_loss_weight 1 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --gpu 1


nice -19 python scripts/train_vid_with_cnn.py \
    --mod lstm_layout \
    --decoder shareContent \
    --obj_size 224,224 \
    --exp AeAlign \
    --encoder copy_image \
    --kl_loss_weight 0 \
    --content_loss_weight 1 \
    --pose_loss_weight 0 \
    --bbox_loss_weight 0 \
    --l1_src_loss_weight 1. \
    --gpu 2
