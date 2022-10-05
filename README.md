##3DSAD: Size Adaptive Clustering for 3D object detection in Point Clouds
 

## Introduction
点云目标检测，baseline工作为VoteNet,运行的基础环境参考自VoteNet
------

 

## Install

Follow VoteNet to install the pointnet++ toolkit and download the dataset. [link](https://github.com/facebookresearch/votenet)

##Data Preparation
We haven't achieved compatibility with the generated data of OpenPCDet yet and use the same data format as mmdeteciton3d for now. We will try to implement indoor data pre-processing based on OpenPCDet as soon as possible.

ScanNet V2
Please install mmdeteciton3d first and follow the data preparation ScanNet V2. Then link the generated data as follows:

ln -s ${mmdet3d_scannet_dir} ./RBGNet/data/scannet
SUN RGB-D
Please install mmdeteciton3d first and follow the data preparation Sun RGB-D. Then link the generated data as follows:

ln -s ${mmdet3d_sunrgbd_dir} ./RBGNet/data/sunrgbd

## Run

cd  3DSAD/ScanNet

​   
​    *change Line2 in train.sh (remove spring.submit; use python ... only)*

sh train.sh



