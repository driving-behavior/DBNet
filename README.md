<img src=https://github.com/driving-behavior/DBNet/blob/master/docs/logo.jpeg width=135/>

[DBNet](http://www.drivingbehavior.org/) is a __large-scale driving behavior dataset__, which provides large-scale __high-quality point clouds__ scanned by Velodyne lasers, __high-resolution videos__ recorded by dashboard cameras and __standard drivers' behaviors__ (vehicle speed, steering angle) collected by real-time censors.

Extensive experiments demonstrate that extra depth information helps networks to determine driving policies indeed. We hope it will become a useful resource for autonomous driving research community.

_Created by Yiping Chen*, Jingkang Wang*, Jonathan Li, Cewu Lu, Zhipeng Luo, HanXue and Cheng Wang. (*equal contribution)_

## Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Contributors](#contributors)
5. [Citation](#citation)
6. [License](#license)

## Introduction
This work is based on our [research paper](https://arxiv.org/abs/1804.xxxx), which is going to appear in CVPR 2018. We propose a large-scale dataset for driving behavior learning, namely, DBNet. You can also check our [dataset webpage](http://www.drivingbehavior.org/) for a deeper introduction.

In this repository, we release __demo code__ and __partial prepared data__ for training with only images, as well as leveraging feature maps or point clouds. The prepared data are accessible [here](https://drive.google.com/open?id=1NjhHwV_q6EMZ6MiGhZnqxg7yRCawx79c). (__More demo models and scripts are released soon!__)

## Requirements

* **Tensorflow 1.2.0**
* Python 2.7
* CUDA 8.0+ (For GPU)
* Python Libraries: numpy, scipy and __laspy__

The code has been tested with Python 2.7, Tensorflow 1.2.0, CUDA 8.0 and cuDNN 5.1 on Ubuntu 14.04. But it may work on more machines (directly or through mini-modification), pull-requests or test report are well welcomed.

## Quick Start
To train a model to predict vehicle speeds and steering angles:

    python train.py --model nvidia_pm --batch_size 16 --gpu 0
The names of the models are consistent with our [paper](http://www.drivingbehavior.org/publications.html).
Log files and network parameters will be saved to `logs` folder in default.

To see HELP for the training script:

    python train.py -h

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir logs

## Contributors
DBNet was developed by [MVIG](http://www.mvig.org/), Shanghai Jiao Tong University* and [SCSC](http://scsc.xmu.edu.cn/) Lab, Xiamen University* (*alphabetical order*).

## Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{DBNet2018,
	  title={DBNet: A Large-Scale Dataset for Driving Behavior Learning},
	  author={Yiping Chen and Jingkang Wang and Jonathan Li and Cewu Lu and Zhipeng Luo and HanXue and Cheng Wang},
	  booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2018}
	  }

## License
Our code is released under Apache 2.0 License. The copyright of DBNet could be checked [here](http://www.drivingbehavior.org).
