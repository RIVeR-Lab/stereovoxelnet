# StereoVoxelNet

![StereoVoxelNet generates voxels from a stereo pair to represent the detected location of the obstacles at the range of 32 meters in a coarse-to-fine manner.](hie.gif)

This is the repositary for paper "Real-Time Obstacle Detection Based on Occupancy Voxels From a Stereo Camera Using Deep Neural Networks". [\[Project Website\]](https://lhy.xyz/stereovoxelnet/)
[\[arXiv\]](https://arxiv.org/abs/2209.08459) [\[Supplementary Video\]](https://www.youtube.com/watch?v=3wju1BbZITM)

## Credits
We would like to thank the generous authors from [MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet), [PSMNet](https://github.com/JiaRenChang/PSMNet), and [GwcNet](https://github.com/xy-guo/GwcNet) for contributing such a great codebase. Several components of our code (dataloader, model components, etc.) are based on their code.

## Installation
The code is tested on following libraries. Libraries with other versions might also work, but not guaranteed:

* Ubuntu 20.04
* Python 3.8
* PyTorch 1.11.0
* CUDA 11.7
* Moviepy 1.0.3 (For visualization)
* ROS Noetic (Optional)

## Dataset

### DrivingStereo
StereoVoxelNet is trained and evaluated using [DrivingStereo](https://drivingstereo-dataset.github.io/) dataset. Please make sure to download their training and testing data and extract to any folder you prefer.

### Our Dataset
We collect a dataset for finetuning. [Our dataset](https://app.globus.org/file-manager?origin_id=09b35740-377b-11ed-89ce-ede5bae4f491&origin_path=%2F) is shared via Globus.

## Evaluation

**Note:** You can try any sample from DrivingStereo dataset. However, evaluating DrivingStereo-trained model with sample from other dataset (KITTI) might lead to downgraded performance.

Please enter the ```visualization``` folder
```
cd scripts/net/visualization/
```

For comparsion with other three approaches, run

```
python visualize_compare.py
```

To visualize the hierarchical output, run

```
python visualize_hie.py
```

To obtain the same result in the paper, like this
| Cost Volume | CD       | IoU |
|:------:|:-------------:|:-------:|
| [48 Levels](https://drive.google.com/file/d/1xK65gfo6tnvfnJB-7Fpflrh6GJ11XePv/view?usp=sharing) | 3.56 | 0.32 |
| [24 Levels](https://drive.google.com/file/d/1J_fTVuw6T8JWeQzK9DMOmAjW_aAoIx36/view?usp=sharing) | 2.96 | 0.34 |
| [12 Levels](https://drive.google.com/file/d/1I87mN5C2MWkc5AWIgrUvtTkB0BJVGcUO/view?usp=sharing) | 3.05 | 0.33 |
| **[Voxel (Ours)](https://drive.google.com/file/d/1zhx4STe9vu6cbQj_jblKLVAQy48yxazh/view?usp=sharing)** | **2.40** | **0.35** |
|

Under the ```./net/``` folder, after changing ```DATAPATH```, ```DATALIST```, and ```ckpt_path```, you can run
```
python test.py
```

## Training (Optional)

To train a model from scratch

```bash
python train.py --dataset voxelds --datapath PATH_TO_DATASET \
 --trainlist PATH_TO_FILENAMES/DS_train.txt \
 --testlist PATH_TO_FILENAMES/DS_test.txt \
 --epochs 20 --lrepochs "10,16:2" --logdir PATH_TO_LOGS/logs \
 --batch_size 4 --test_batch_size 4 --summary_freq 50 --loader_workers 8 \
 --cost_vol_type voxel --model Voxel2D
```


## Citation

If you use this code, please cite this paper:  

```
@inproceedings{li2023stereovoxelnet,
title = {StereoVoxelNet: Real-Time Obstacle Detection Based on Occupancy Voxels from a Stereo Camera Using Deep Neural Networks},
author = {Li, Hongyu and Li, Zhengang and Akmandor, Neset Unver and Jiang, Huaizu and Wang, Yanzhi and Padir, Taskin},
booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
year={2023}
}
```