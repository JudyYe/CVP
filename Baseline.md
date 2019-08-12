## Baselines Configurations
We have provided code to reimplement baselines to ablate predictor, decoder, and encoder correpondingly.

The default configuration is set to train our CVP:
```angular2
python train.py --gpu ${GPU_ID} --mod cvp --decoder comb --encoder traj
```

### Entity Predictor
- No-Factor: [Generating Videos with Scene Dynamics](https://papers.nips.cc/paper/6194-generating-videos-with-scene-dynamics), Vondrick et al. in NeurIPS'16
```angular2
python train.py --gpu ${GPU_ID} --mod noFactor  --graph fact_fc --decoder noFactor
```
- No-Edge:
```angular2
python train.py --gpu ${GPU_ID} --gconv_unit_type noEdge
```

### Frame Decoder
- Early-Feat
```
python train.py --dec_dims 256,128,64,32
```
- Mid-Feat
```
python train.py --dec_dims 256,128,64,32,16
```
- Late-Feat (ours)
```
python train.py --dec_dims 256,128,64,32,16,8
```
- Pixel
```
python train.py --decoder cPix
```

### Latent Representations and Encoder
The `--encoder` could be `traj`(ours), `noZ`, `fp`, or `lp`.

- No-Z baseline
``` 
python train.py --encoder noZ
```
- FP (Fix-Prior): [Stochastic Video Generation with a Learned Prior](https://arxiv.org/pdf/1802.07687.pdf), Denton et al. in ICML'18
```
python train.py --encoder fp
```
- LP (Learned-Prior): [Stochastic Video Generation with a Learned Prior](https://arxiv.org/pdf/1802.07687.pdf), Denton et al. in ICML'18
```
python train.py --mod lp --encoder lp
```


## Testing
All models could directly be evaluated by running:
```angular2
python test.py --checkpoint ${PATH_TO_MODEL} --test_mod best_100
```
