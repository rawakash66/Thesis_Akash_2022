# Computer Vision Based Liquid Contact Angle Estimation from 3D Reconstructed Droplets

## Table of contents
[Introduction](#introduction)
[Installation](#installation)

## Introduction <a name="introduction"></a>
This repository includes the code scripts implemnted in the thesis titled ''Computer Vision Based Liquid Contact Angle Estimation from 3D Reconstructed Droplets'' submitted in Indian Institute of Technology, Kanpur for the partial fulfilment of the requirements for the degree of Master of Technology. Towards the broad goal of reconstruction of a transparent object, this method presents a novel approach for 3D reconstruction of static liquid droplets on smooth, homogenous surfaces. The following studies were performed. First, a high resolution mobile camera equipped with macro lens is used for image acquisition of small-sized droplet. Next, for estimating intrinsic and extrinsic camera parameters a printed pattern was used. After that, U-Net CNN architecture was used to extract silhouettes of droplets from digital images using semantic segmentation. Finally, shape-from-silhouette method was employed with space carving algorithm to estimate the visual hull containing the droplet volume. The following is the flowdiagram of the methodology.

<p align="center">
  <img src="https://github.com/rawakash66/Thesis_Akash_2022/blob/main/Carving%20methodology.png" width="600">
</p>

## Installation <a name="intstallation"></a>
### Requirements
1. python3
2. tensorflow-gpu
3. opencv-gpu
4. numpy
5. matplotlib

For installing OpenCV with GPU support follow the instructions provided in this [video](https://www.youtube.com/watch?v=HsuKxjQhFU0). All other libraries can be installed after this by executing the below code.
