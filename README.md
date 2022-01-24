# Decoding Depth to RGB
RGB Image Reconstruction from Depth Data using AutoEncoder and Object Segmentation

![image](https://user-images.githubusercontent.com/67869508/150832590-b411b1fb-98f4-487e-9542-00af6f861f71.png)

## Abstract

This experiments is a preliminary study on real-time instance segmentation in an lightless indoor space using only the depth image obtained by the ToF RGB-D Camera. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/67869508/136656185-a714f9ec-1674-432d-821f-0b454d594bab.png">
  <br> [Mapping Depth Channel to RGB Channel(8bit) - Depth Visualization and Preprocessing for uses like RGB Images]
</p>

Paper : **Depth-based Instance Segmentation for Object Recognition in Lightless Indoor Space, KISM`21 Fall** <br>
[BumSoo Kim](https://github.com/gh-BumsooKim)\*, Graphics Realization Lab, CAU, (* CAU A&T)

## Experimental Environments

- Windows10
- Microsoft Azure Kinect (ToF)
- YOLACT Model (Backbone : Resnet101-FPN)
- NYUv2 Dataset for Training

##  Training Dataset

- using **NYUv2 (NYU-Depth Dataset V2), [Introduced by Nathan Silberman et al. in Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf)**

![](http://cs.nyu.edu/~silberman/images/nyu_depth_v2_web.jpg)
<br> \[Indoor Segmentation and Support Inference from RGBD Images, Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus, ECCV 2012\]
  
## Testing Model

- using **Yolact (You Only Look At CoefficienTs), [A simple, fully convolutional model for real-time instance segmentation](https://github.com/dbolya/yolact)**
