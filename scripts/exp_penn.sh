#!/usr/bin/env bash


bash scripts/bash_data.sh 2 pennAugRectPullUdE_fair_skipDst_C0P32Z8_onlycell_image_fastSig7658batch_fact_gc_noEdge_Ppix1_11No_dt8 200 all RectPull
bash scripts/bash_data.sh 1 pennAugRectPullFC_fair_skipDst_bs_C0P32Z8_onlycell_image_softNoSp7658batch_fact_fc_n2e2n_Ppix1_11No_dt8 200 all RectPull

--ss3 SoftNoSp

python scripts/train_vid_zero_grad.py  \
    --dataset ss3  --epoches 530 --mod skipDst_bs \
    --decoder softNoSp \
    --graph fact_fc \
    --exp fairPredBug \
    --gpu 1

python scripts/train_vid_zero_grad.py  \
    --dataset ss3  --epoches 530 --mod skipDst_bs \
    --decoder skipNoSp \
    --graph fact_fc \
    --exp fairPredBug \
    --gpu 1


 --crop Gym

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fairBack \
    --gpu 1


python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fair \
    --gpu 1


# no spatial
python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --dt 8 --bs 8 --epoches 630 --mod skipDst_bs \
    --modality FC \
    --decoder softNoSp \
    --exp fair \
    --graph fact_fc \
    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --dt 8 --bs 8 --epoches 630 --mod skipDst_bs \
    --modality FC \
    --decoder skipNoSp \
    --exp fair \
    --graph fact_fc \
    --gpu 1


python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fair \
    --gconv_unit_type noEdge \
    --gpu 1


#python scripts/train_vid_zero_grad.py  --mod skipDst \
#    --dataset pennAugGym --modality All --dt 8 --bs 8 --epoches 630 \
#    --exp fair \
#    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectPull --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fairBack \
    --gpu 1


# Pull Up
python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectPull --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fair \
    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectPull --modality UdE --dt 8 --bs 8 --epoches 630 \
    --exp fair \
    --gconv_unit_type noEdge \
    --gpu 1


python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectPull --dt 8 --bs 8 --epoches 630 --mod skipDst_bs \
    --modality FC \
    --decoder softNoSp \
    --exp fair \
    --graph fact_fc \
    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugRectPull --dt 8 --bs 8 --epoches 630 --mod skipDst_bs \
    --modality FC \
    --decoder skipNoSp \
    --exp fair \
    --graph fact_fc \
    --gpu 1


-- no bbox again?
python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset ss3   --epoches 530 \
    --l1_src_loss_weight 0 \
    --bbox_loss_weight 0 \
    --exp unsupBoxRecon \
    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset ss3   --epoches 530 \
    --bbox_loss_weight 0 \
    --exp unsupBox \
    --gpu 1

--soft bn

python scripts/train_vid_zero_grad.py  \
    --dataset ss3  --epoches 530 --mod skipDst_bs \
    --decoder softNoSp \
    --graph fact_fc \
    --exp fairPred \
    --gpu 1


python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugGym --dt 8 --bs 8 --epoches 630 --mod skipDst_bs \
    --modality FC \
    --decoder softNoSp \
    --exp fair \
    --graph fact_fc \
    --gpu 1

python scripts/train_vid_zero_grad.py  --mod skipDst \
    --dataset pennAugPul --dt 8 --bs 8 --epoches 630 --mod skipDst_bs \
    --modality FC \
    --decoder softNoSp \
    --exp fair \
    --graph fact_fc \
    --gpu 1

---

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder onlycell_image \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrdback \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder copy_image \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder iid_image \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder lstm_lp \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

--predictor
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder onlycell_image \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gconv_unit_type noEdge \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 \
    --encoder onlycell_image \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gconv_unit_type node \
    --gpu 1

#python scripts/train_vid_zero_grad.py \
#    --dataset ss3  --epoches 530  --noise_dim 8 \
#    --encoder onlycell_image \
#    --decoder skipNoSp  --graph fact_fc --mod skip_bs \
#    --exp boxOrd \
#    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530  --noise_dim 8 \
    --encoder onlycell_image \
    --graph fact_in  \
    --exp boxOrd \
    --gpu 1

--decoder
python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --encoder onlycell_image \
    --decoder fastSig \
    --dec_dims 256,128,64,32,16 \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --encoder onlycell_image \
    --decoder fastSig \
    --dec_dims 256,128,64,32 \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

#python scripts/train_vid_zero_grad.py \
#    --dataset ss3   --epoches 530 \
#    --encoder onlycell_image \
#    --decoder fastSig \
#    --noise_dim 8 \
#    --decoder pixBnRelu \
#    --exp boxOrd \
#    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --encoder onlycell_image \
    --decoder fastSig \
    --noise_dim 8 \
    --decoder pixSig \
    --exp boxOrd \
    --gpu 1

--
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

--PREDICTOR
python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gconv_unit_type noEdge \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriGym --modality All --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriGym --dt 8 --bs 8 --epoches 630 \
    --modality FC \
    --decoder skipNoSp   --graph fact_fc --mod skip_bs\
    --exp boxOrd \
    --gpu 1

--other datset
python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriPull --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriPush --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriSit --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriJack --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1


--
python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriTenn --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriBase --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriBowl --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriGolf --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriOut --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriSquat --modality UdE --dt 8 --bs 8 --epoches 730 \
    --encoder onlycell_image \
    --decoder fastSig \
    --exp boxOrd \
    --gpu 1



--ss3 box?? encoder , noEdge

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --dec_skip 7 \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --dec_skip 7 \
    --encoder copy_res \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --dec_skip 7 \
    --encoder iid_res \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --dec_skip 7 \
    --encoder lstm_lp \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --dec_skip 7 \
    --decoder fastSig \
    --noise_dim 8 \
    --exp boxOrd \
    --gconv_unit_type noEdge \
    --gpu 1

- decoder
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --epoches 530 --dec_skip 7 \
    --decoder pixBnRelu \
    --noise_dim 8 \
    --exp boxOrd \
    --gpu 1

-penn?
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --decoder fastSig \
    --exp blue \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennAugOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --decoder fastSig \
    --exp blue \
    --gpu 1

--ss3
python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --gconv_unit_type noEdge \
    --decoder sigOne \
    --noise_dim 8 \
    --exp goodNoEdge \
    --gpu 1

--penn why no shareBG?
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder act_hid \
    --decoder shareBg --mod bg\
    --exp whynot \
    --gpu 1



--fall css??
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --mod bg --epoches 530 \
    --decoder shareBg \
    --noise_dim 8 \
    --dec_zero_grad 1 \
    --exp fallCss \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --mod bg --epoches 530 \
    --decoder shareBg \
    --noise_dim 8 \
    --l1_dst_loss_weight 10 \
    --exp fallCss \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --mod bg --epoches 530 \
    --decoder shareBg \
    --noise_dim 8 \
    --l1_dst_loss_weight 10 \
    --dec_zero_grad 1 \
    --exp fallCss \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --dec_skip 7 --epoches 530 \
    --decoder fastSig \
    --noise_dim 8 \
    --exp fallCss \
    --gpu 1

--ss3 decoder
decoder: sigOne	?
bgShare	?

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --decoder sigOne \
    --noise_dim 8 \
    --exp badDec \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --mod bg --epoches 530 \
    --decoder shareBg \
    --noise_dim 8 \
    --exp badDec \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --mod bg --epoches 530 \
    --decoder shareBg \
    --noise_dim 8 \
    --l1_dst_loss_weight 10 \
    --exp badDec \
    --gpu 1


--ss3 encoder only box
python scripts/train_traj.py \
    --dataset ssbox --mod lstm_box  --epoches 530 \
    --encoder box_lstm \
    --pose_dim 1 --content_dim  3 \
    --exp traj \
    --gpu 1

python scripts/train_traj.py \
    --dataset ssbox --mod lstm_box  --epoches 530 \
    --encoder box_copy \
    --pose_dim 1 --content_dim  3 \
    --exp traj \
    --gpu 1

python scripts/train_traj.py \
    --dataset ssbox --mod lstm_box  --epoches 530 \
    --encoder box_lstm \
    --pose_dim 1 --content_dim  3 \
    --noise_dim 8 \
    --exp traj \
    --gpu 1

python scripts/train_traj.py \
    --dataset ssbox --mod lstm_box  --epoches 530 \
    --encoder box_copy \
    --pose_dim 1 --content_dim  3 \
    --noise_dim 8 \
    --exp traj \
    --gpu 1

--ss3 encoder

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --noise_dim 8 \
    --encoder copy_res \
    --decoder shareBg --mod bg \
    --exp goodEnc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --noise_dim 8 \
    --exp goodEnc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --noise_dim 8 \
    --encoder copy_res \
    --exp goodEnc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --noise_dim 8 --content_dim 128\
    --exp goodEnc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3   --epoches 530 \
    --noise_dim 8 --content_dim 128\
    --encoder copy_res \
    --exp goodEnc \
    --gpu 1


--penn content pose

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --bs 32 --epoches 530 \
    --encoder onlycell_image \
    --decoder bnRelu \
    --dec_zero_grad 0 \
    --exp zero \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder bnRelu \
    --content_dim 32 --pose_dim 8 \
    --dec_zero_grad 0 \
    --exp zero \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder bnRelu \
    --dec_zero_grad 0 \
    --exp zero \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder bnRelu \
    --bbox_loss_weight 1000 \
    --dec_zero_grad 0 \
    --exp zero_BB3 \
    --gpu 1



-

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder act_hid \
    --decoder bnRelu \
    --content_dim 32 --pose_dim 8 \
    --exp cp \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --encoder onlycell_image \
    --decoder bnRelu \
    --content_dim 32 --pose_dim 8 \
    --exp cp \
    --gpu 1


--
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --decoder bnRelu \
    --exp cp \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --decoder bnRelu \
    --content_dim 32 --pose_dim 8 \
    --exp cp \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 630 \
    --decoder bnRelu \
    --content_dim 32 --pose_dim 8 \
    --strip 4 \
    --exp cp_strip4 \
    --gpu 1
--encoder


python scripts/train_vid_zero_grad.py \
    --dataset ss3 --bs 32 --epoches 530 \
    --mod bg \
    --noise_dim 8 \
    --encoder onlycell_image \
    --decoder shareBg \
    --l1_dst_loss_weight 10 \
    --exp enc_bs32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --bs 32 --epoches 530 \
    --mod bg \
    --content_dim 128 --noise_dim 8 \
    --encoder onlycell_image \
    --decoder shareBg \
    --l1_dst_loss_weight 10 \
    --exp enc_bs32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --mod bg \
    --content_dim 128 --noise_dim 8 \
    --encoder onlycell_image \
    --decoder shareBg \
    --l1_dst_loss_weight 10 \
    --exp enc \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --content_dim 128 --noise_dim 8 \
    --encoder onlycell_image \
    --l1_dst_loss_weight 10 \
    --exp enc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --content_dim 128 --noise_dim 8 \
    --encoder onlycell_image \
    --decoder bnRelu \
    --l1_dst_loss_weight 10 \
    --exp enc \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --content_dim 128 --noise_dim 8 \
    --l1_dst_loss_weight 10 \
    --exp enc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --content_dim 128 --noise_dim 8 \
    --exp enc \
    --gpu 1

-- why the strange color?
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --l1_dst_loss_weight 10 \
    --exp color_dst10 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --content_dim 32 \
    --noise_dim 8 \
    --exp color \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --noise_dim 8 \
    --exp color \
    --gpu 1

--
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --decoder bnRelu \
    --exp spider \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 \
    --decoder bnRelu \
    --gconv_unit_type noEdge \
    --exp spider \
    --gpu 1



--BN???
# less pose
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder allBn  \
    --dec_norm bnrs \
    --pose_dim 32 \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder allBn  \
    --dec_norm bnrs \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder allBn  \
    --dec_norm bnaf \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder allBn  \
    --dec_norm instance \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder allBn  \
    --dec_norm bnff \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1
-
# align
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm bnrs \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm bnaf \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm instance \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm bnff \
    --l1_dst_loss_weight 1 \
    --exp bn \
    --gpu 1


--double old
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp repro \
    --modality UdE \
    --decoder oldOne \
    --dec_norm none \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp repro \
    --modality UdE \
    --decoder oldOne \
    --dec_norm none \
    --l1_dst_loss_weight 1 \
    --gpu 1



 --double check
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp doubleCheckNoBN \
    --modality UdE \
    --decoder regOne \
    --dec_dims 256,128,64,32,16,8 \
    --dec_norm none \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
   --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
   --exp doubleCheck \
   --modality UdE \
   --decoder regOne \
   --dec_dims 256,128,64,32,16,8 \
   --kl_loss_weight 0 \
   --pose_pix_loss False \
   --bbox_loss_weight 0 \
   --gpu 1


python scripts/train_vid_zero_grad.py \
   --dataset ss3 --dt 16 --bs 16 --epoches 530 --mod skip\
   --exp doubleCheck \
   --decoder regOne \
   --dec_dims 256,128,64,32,16,8 \
   --dec_norm none \
   --kl_loss_weight 0 \
   --pose_pix_loss False \
   --bbox_loss_weight 0 \
   --gpu 1


--masknet



python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --gpu 1


-- yanzhe

# no bn??
# this should give us amazing real_recon
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder bnOne  \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --exp yanzhe \
    --gpu 1

# if this give us amazing real_recon, we should try all efforts to make ss3 work!!
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder bnOne  \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1
# this should possibly collaps...
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder bnOne  \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1



python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder bnOne  \
    --dec_norm batch \
    --gconv_normalization batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder bnOne  \
    --dec_norm batch \
    --gconv_normalization batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --gconv_normalization batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --gconv_normalization batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1
--





python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder maskOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder maskOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --exp yanzheAE \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp yanzhe \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --exp yanzheAE \
    --gpu 1




-
python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder maskOne  \
    --l1_dst_loss_weight 1 \
    --exp maskNet-10 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder maskOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp maskNet-10 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --l1_dst_loss_weight 1 \
    --exp maskNet-10 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp maskNet-10 \
    --gpu 1


--


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder softOne  \
    --dec_norm batch \
    --l1_dst_loss_weight 1 \
    --exp strange \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder sigOne  \
    --l1_dst_loss_weight 1 \
    --exp strange-1 \
    --gpu 1



python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder softOne \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --exp strangeAE \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym  --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --decoder softOne \
    --l1_dst_loss_weight 1 \
    --exp strange \
    --gpu 1





--BN debug

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder softOne  \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder softOne  \
    --exp mask-100 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --decoder softOne  \
    --exp debugBG-1 \
    --l1_dst_loss_weight 1 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --dec_norm batch \
    --exp debugBG \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --exp debugBG \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --exp debugBGMASK \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --exp debugBGMASK \
    --l1_dst_loss_weight 1 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --dec_skip 7 \
    --exp debugBG \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --dec_skip 7,6 \
    --exp debugBG \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --dec_skip 7,6,5,4 \
    --exp debugBG \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp debugBGAE \
    --modality UdE \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp debugBGAE \
    --modality UdE \
    --dec_norm batch \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp debugBGAE \
    --modality UdE \
    --dec_skip 7 \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp debugBG \
    --modality UdE \
    --dec_skip 7 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp debugBGAE \
    --modality UdE \
    --dec_skip 7,6 \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp debugBGAE \
    --modality UdE \
    --dec_norm batch \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1



-- No BN, default to dims...,16,8, pose_dim 128

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --exp NoBN \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --exp NoBN \
    --l1_dst_loss_weight 1 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --exp NoBNBB \
    --bbox_loss_weight 1000 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip \
    --exp NoBNBB \
    --l1_dst_loss_weight 1 \
    --bbox_loss_weight 1000 \
    --gpu 1


-

python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --exp NoBN \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3  --bs 16 --epoches 530 --mod skip \
    --exp NoBN \
    --l1_dst_loss_weight 1 \
    --gpu 1
-
python scripts/train_vid_zero_grad.py \
    --dataset ss3 --bs 16 --epoches 530 --mod skip_bs \
    --exp NoBN \
    --decoder skipNoSp \
    --graph fact_fc \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --bs 16 --epoches 530 --mod skip_bs \
    --exp NoBN \
    --l1_dst_loss_weight 1 \
    --decoder skipNoSp \
    --graph fact_fc \
    --gpu 1


--learned prior ss3
python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip_lp --bs 16\
    --exp full \
    --encoder lstm_lp \
    --decoder maskSkipSplat \
    --gpu 1



--penn
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp full1k \
    --bbox_loss_weight 1000 \
    --decoder maskSkipSplat \
    --gpu 1
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp full76541k \
    --box_loss_weight 1000 \
    --dec_dims 7,6,5,4 \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp full \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp full \
    --encoder onlycell_image \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp full7654 \
    --dec_dims 7,6,5,4 \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --modality UdE --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp full7654 \
    --encoder onlycell_image \
    --dec_dims 7,6,5,4 \
    --decoder maskSkipSplat \
    --gpu 1

--ss3

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew32 \
    --dec_dims 256,128,64,32 \
    --decoder regOne \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew16 \
    --dec_dims 256,128,64,32,16 \
    --decoder regOne \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew8 \
    --dec_dims 256,128,64,32,16,8 \
    --decoder regOne \
    --gpu 1



python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew32 \
    --dec_dims 256,128,64,32 \
    --decoder regSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew16 \
    --dec_dims 256,128,64,32,16 \
    --decoder regSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew8 \
    --dec_dims 256,128,64,32,16,8 \
    --decoder regSkipSplat \
    --gpu 1




python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip\
    --exp full \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew16 \
    --dec_dims 256,128,64,32,16 \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp fullNew8 \
    --dec_dims 256,128,64,32,16,8 \
    --decoder maskSkipSplat \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip --bs 16\
    --exp full16 \
    --dec_dims 256,128,64,32,16 \
    --encoder onlycell_image \
    --decoder maskSkipSplat \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset ss3 --epoches 100 --mod skip\
    --exp full \
    --decoder fancySkip \
    --gpu 1


--test ae, share background

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp aeNew8NoBN \
    --modality UdE \
    --decoder regOne \
    --dec_dims 256,128,64,32,16,8 \
    --dec_norm none \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp aeNew8 \
    --modality UdE \
    --decoder regOne \
    --dec_dims 256,128,64,32,16,8 \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ae_16New \
    --modality UdE \
    --decoder maskSkipSplat \
    --dec_dims 256,128,64,32,16 \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ae765_8 \
    --modality UdE \
    --decoder maskSkipSplat \
    --dec_dims 128,64,32,16,8,8 \
    --gconv_num_blocks 2 \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ae7654 \
    --modality UdE \
    --dec_skip 7,6,5,4 \
    --decoder maskSkipSplat \
    --kl_loss_weight 0 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


-- test ae, no share background
python scripts/train_only_ae.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod ae\
    --exp ae765 \
    --modality UdE \
    --decoder aeSkip \
    --gpu 1

    python scripts/train_only_ae.py \
    --dataset SynBlock --dt 8 --bs 8 --epoches 530 --mod ae\
    --exp tmp \
    --modality 01 \
    --dec_dims 256,128,64,32 \
    --dec_skip 7,6,5,4 \
    --decoder aeSkip \
    --gpu 1

python scripts/train_only_ae.py \
    --dataset SynBlock --dt 8 --bs 8 --epoches 530 --mod ae\
    --exp tmp \
    --modality 255 \
    --dec_dims 256,128,64,32 \
    --dec_skip 7,6,5,4 \
    --decoder aeSkip \
    --gpu 1



-- test decoder splat mask + feat
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splatAE7654 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --dec_skip 7,6,5,4 \
    --decoder maskSkipSplat \
    --encoder onlycell_res \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 3 --bs 3 --epoches 530 --mod skip\
    --exp splatAE7654 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --dec_skip 7,6,5,4 \
    --decoder maskSkipSplat \
    --encoder onlycell_res \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splatAE \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder maskSkipSplat \
    --encoder onlycell_res \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 3 --bs 3 --epoches 530 --mod skip\
    --exp splatAE \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder maskSkipSplat \
    --encoder onlycell_res \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splatAE \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder maskSkipSplat \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --gpu 1


--test encoder res
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splat \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipSplat \
    --encoder onlycell_res \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splat \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --encoder onlycell_res \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

---Splat
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splat \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipSplat \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splat \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipSplat \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splatAE \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipSplat \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp splatAE \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipSplat \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

-- No bbox
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skipDst\
    --exp noMed \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skipDst\
    --exp noMed \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skipDst\
    --exp noBBox5 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skipDst\
    --exp noBBox5 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skipDst\
    --exp noBBox \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skipDst\
    --exp noBBox \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


--- Less
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minhBbox \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --bbox_loss_weight 1000 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1



python scripts/train_vid_zero_grad.py \
    --dataset pennOriLess --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1



---Skip
python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh2 \
    --dec_skip 7,6 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh2 \
    --dec_skip 7,6 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh1 \
    --dec_skip 7 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh1 \
    --dec_skip 7 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


---crop
python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp minh \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

--Ae
python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ae \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ae \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ae \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --pose_pix_loss False \
    --bbox_loss_weight 0 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1
---

python scripts/train_vid_zero_grad.py \
    --dataset pennRdPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1
---
python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 32 --pose_dim 32 \
    --gpu 1
---

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriBase --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkipHigh \
    --content_dim 0 --pose_dim 32 \
    --gpu 1
---
python scripts/train_vid_zero_grad.py \
    --dataset pennOriBase --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder keySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


# foreground
python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

--

python scripts/train_vid_zero_grad.py \
    --dataset pennOriBase --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder shareBg \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriTenn --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder shareBg \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder shareBg \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder shareBg \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

--

python scripts/train_vid_zero_grad.py \
    --dataset pennOriBase --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fgSkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriTenn --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fgSkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fgSkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriPull --dt 8 --bs 8 --epoches 530 --mod bg_fore\
    --exp foreC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fgSkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

----
# multi class
python scripts/train_vid_zero_grad.py \
    --dataset pennOriBase --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip_bs\
    --modality UdE \
    --decoder skipNoSp \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennOriGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1
--

python scripts/train_vid_zero_grad.py \
    --dataset pennRdBase --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdGym --dt 8 --bs 8 --epoches 530 --mod skip_bs\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder skipNoSp \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovftC0 \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

----
#multi class for ovft
python scripts/train_vid_zero_grad.py \
    --dataset pennMcBaseTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennMcGym --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennMcAll --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennMcTenS --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdTenS --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennRdTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennMcTenn --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennMcBase --dt 8 --bs 8 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1
----
# Solve overfitting
python scripts/train_vid_zero_grad.py \
    --dataset pennCropPart --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp ovft \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropPart --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp ovft \
    --modality FcE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

---

# exp after fixing bugs of dt, v -> N, not v, dt -> N
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality DrE \
    --dec_dims 256,128,64,32 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropLimb --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality DrE \
    --dec_dims 256,128,64,32 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropLimb --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

--
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality DrE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


******************************
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1
******************************



python scripts/train_vid_zero_grad.py \
    --dataset pennCropLimb --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality DrE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropLimb --dt 8 --bs 16 --epoches 530 --mod skip\
    --exp dtV \
    --modality UdE \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1




---
# exp on background skip, default on (joint, direct Edge)
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp back \
    --modality DrE \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp backLessDim \
    --modality DrE \
    --mod skip \
    --dec_dims 256,128,64,32 \
    --decoder fancySkip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1
-----

# only loss on box, but with joint. Two side (some directed edge!)
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality DrE \
    --mod loc \
    --content_dim 32 --pose_dim 32 \
    --pose_pix_loss False \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality DrE \
    --mod bg_fore \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality DrE \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

-
# only loss on box

python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality DrE \
    --mod loc \
    --content_dim 32 --pose_dim 32 \
    --pose_pix_loss False \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality DrE \
    --mod bg_fore \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality DrE \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

---
# only loss on box, but with joint. Two side (undirected edge!)
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality UdE \
    --mod loc \
    --content_dim 32 --pose_dim 32 \
    --pose_pix_loss False \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality UdE \
    --mod bg_fore \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality UdE \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

-
# only loss on box

python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality UdE \
    --mod loc \
    --content_dim 32 --pose_dim 32 \
    --pose_pix_loss False \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality UdE \
    --mod bg_fore \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp edge \
    --modality UdE \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


--
# only loss on box, but with joint. oh. No. these skeleton are just one side!!!
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp onlyLoc \
    --mod loc \
    --content_dim 32 --pose_dim 32 \
    --pose_pix_loss False \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp onlyFore \
    --mod bg_fore \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp skip \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

--
# only loss on box

python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp onlyLoc \
    --mod loc \
    --content_dim 32 --pose_dim 32 \
    --pose_pix_loss False \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp onlyFore \
    --mod bg_fore \
    --content_dim 32 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --exp skip \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


--
python scripts/train_vid_zero_grad.py \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --exp skip \
    --mod skip \
    --dec_dims 512,256,128,64 \
    --decoder skip \
    --content_dim 32 --pose_dim 32 \
    --gpu 1


python scripts/train_vid_zero_grad.py \
    --exp skip \
    --dataset pennCropJoint --dt 8 --bs 16 --epoches 530 \
    --decoder RegBg \
    --content_dim 32 --pose_dim 32 \
    --gpu 1
--
python scripts/train_vid_zero_grad.py \
    --exp test \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1

python scripts/train_vid_zero_grad.py \
    --exp testAe \
    --dataset pennCrop --dt 8 --bs 16 --epoches 530 \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --content_loss_weight 0 \
    --bbox_loss_weight 0 \
    --gpu 1
--
original penn action

python scripts/train_vid_zero_grad.py \
    --exp test \
    --dataset penn --dt 8 --bs 16 --epoches 530 \
    --content_dim 0 --pose_dim 32 \
    --gpu 1


--
#only ae
python scripts/train_vid_zero_grad.py \
    --exp testAe \
    --dataset penn --dt 8 --bs 16 --epoches 530 \
    --content_dim 0 --pose_dim 32 \
    --pose_pix_loss False \
    --content_loss_weight 0 \
    --bbox_loss_weight 0 \
    --gpu 1