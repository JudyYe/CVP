# Dataset Preparation

## [Shapestacks](https://shapestacks.robots.ox.ac.uk)
Shapestacks provides the initial RGB images and configuration files. We render videos via MuJoCo for 32 steps.  

### Format
Each video includes:
```
cam_1.npy: (x,y) of object center in image coordinate with scale of [0, 1], in shape of (T, O, 2). T is the length of the videos, O is the nubmer of objects. 
rgb-cam_1-00.jpg: RGB images in shape of (244, 244, 3)
... 
rgb-cam_1-31.jpg
```

### Code to render
Coming soon!

## [PennAction](https://dreamdragon.github.io/PennAction/)
Coming soon!