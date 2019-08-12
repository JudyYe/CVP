#!/usr/bin/env bash


# to CVP with default config
python train.py

# Ablate Predictor
#   No-Factor baseline: Generating Videos with Scene Dynamics, Vondrick et al. in NeurIPS'16
python train.py --mod noFactor    --graph fact_fc --decoder noFactor

#   No-Edge baseline
python train.py     --gconv_unit_type noEdge

# Ablate Encoder
# No-Z baseline
python train.py --encoder noZ

# FP (Fix-Prior): Stochastic Video Generation with a Learned Prior, Denton et al. in ICML'18
python train.py --encoder fp

# LP (Learned-Prior): Stochastic Video Generation with a Learned Prior, Denton et al. in ICML'18
python train.py --encoder lp

# Ablate Decoder
# Early-Feat
python train.py --dec_dims 256,128,64,32

# Mid-Feat
python train.py --dec_dims 256,128,64,32,16

# Pixel
python train.py --decoder cPix

