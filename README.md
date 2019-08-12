# Compositional Video Prediction
Yufei Ye,    Maneesh Singh,    Abhinav Gupta*, and   Shubham Tulsiani*
   
[Project Page](https://judyye.github.io/CVP/), [Arxiv]() 
![](docs/pred1.gif)

This code is a re-implementation of the paper [Compositional Video Prediciton](arxive link). The code is developed based on Pytorch framework.

Given an initial frame, the task is to predict the next few frames in pixel level. The key insight is that a scene is comprised of distinct entities that undergo joint motions.
To operationalize this idea, we propose **Compositional Video Prediction** (CVP), which consists of three main modules:
1) **Entity Predictor**: predicts per-entity representation;
2) **Frame Decoder**: generate pixels given entity-level representation;
3) **Encoder**: generate latent variables to account for multi-modality.     

![](docs/pipeline.png)


## Citation
If you find this work useful, please use the following BibTeX entry.

```
@article{ye2019cvp,
  title={Compositional Video Prediction},
  author={Ye, Yufei and Singh, Maneesh  and Gupta, Abhinav and Tulsiani, Shubham},
  year={2019},
  booktitle={International Conference on Computer Vision (ICCV)}
}
```

## Setup Repo
The code was developed by Python 3.6 and PyTorch 0.4.

`git clone git@github.com:JudyYe/CVP.git`



## Demo: Predict video with pretrained model
1. Download pretrained model [here](google drive) and put them to `models/`.   

2. Use our models to predict videos for each image under `examples/`. This generates 5 random sampled videos. Each row corresponds to one sample. 
```
python demo.py --checkpoint ${MODEL_PATH} --test_mod multi
``` 


## Set up Dataset
To quantitatively evaluate or to train your own models, you need to set up dataset first. 
In the paper, results on two datasets are provided: the synthetic dataset [Shapestacks](https://shapestacks.robots.ox.ac.uk) and [PennAction](https://dreamdragon.github.io/PennAction/).  

For a quick setup of ready-to-go data for Shapestacks, download and link to `data/shapestacks/`
```
 cd ${FOLDER_TO_SAVE_DATA}
 wget -O ss3456_render.tar.gz -L https://cmu.box.com/shared/static/fhv9b3nojecys5d2sprlyzi2xv8gjqzt.gz && tar xzf ss3456_render.tar.gz 
 ln -s ${FOLDER_TO_SAVE_DATA}/shapestacks data/shapestacks
```  
 

Please read [`Dataset.md`](Dataset.md) for data format together with how to generate and preprocess the data.
  

## Quantitative Evaluation
The best scores among K (K=100) samples are recorded. See paper for further explanation. 
(The quality of frame is evaluated based on code repo [LPIPS](https://github.com/richzhang/PerceptualSimilarity))
```
python test.py --checkpoint ${PATH_TO_MODEL} --test_mod best_100 --dataset ss3
```
The models are trained with 3 blocks in Shapestacks. Substitute `ss3` with `ss4`  (or `ss5`, `ss6`) to evaluate how model generalizes to more blocks: 
```
python test.py --checkpoint ${PATH_TO_MODEL} --test_mod best_100 --dataset ss4
```

## Train your own model
The model and logs will be saved to `output/`. To train our model, simply run 
```angular2
python train.py --gpu ${GPU_ID}
```

We have provided code to reimplement baselines to ablate predictor, decoder, and encoder correspondingly.
Please see [`Baseline.md`](Baseline.md) for further details. 
 
