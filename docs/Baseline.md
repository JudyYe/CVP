## Baselines Configurations
We have provided code to reimplement baselines to ablate predictor, decoder, and encoder correpondingly.


The baselines are configured as the following:

 method | mod | graph | gconv_unit_type | encoder | decoder 
---:|:---: | :---: | :---: | :---: | :---:
CVP (default) |  cvp | fact_gc | n2e2n | traj | comb_late
No-Factor [[23]]((https://papers.nips.cc/paper/6194-generating-videos-with-scene-dynamics)) | **noFactor** | **fact_fc** | - | traj | **noFactor** 
No-Edge | cvp | fact_gc | **noEdge** | traj | comb_late
Early-Feat | cvp | fact_gc | n2e2n | traj | **comb_early**
Mid-Feat | cvp | fact_gc | n2e2n | traj | **comb_mid**
Pixel | cvp | fact_gc | n2e2n | traj | **cPix**
No-Z  | cvp | fact_gc | n2e2n | **noZ**| comb_late
FP [[6]](https://arxiv.org/pdf/1802.07687.pdf)  | cvp | fact_gc | n2e2n | **fp**| comb_late
LP [[6]](https://arxiv.org/pdf/1802.07687.pdf)  | **lp** | fact_gc | n2e2n | **lp**| comb_late

To run baseline, just set the non-default flag (in bold) to the corresponding one, since the default configuration is set to train our CVP.

Anyway, the remaining part kindly provides straightforward command which gives you the same config as the table shows:
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
python train.py --decoder comb_early
```
- Mid-Feat
```
python train.py --decoder comb_mid
```
- Late-Feat (ours)
```
python train.py --decoder comb_late
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

