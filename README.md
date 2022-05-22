# Computer Vision Based Liquid Contact Angle Estimation from 3D Reconstructed Droplets

## Table of contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Camera calibration](#calibration)

## Introduction <a name="introduction"></a>
This repository includes the data and code scripts utilized in the thesis titled ''Computer Vision Based Liquid Contact Angle Estimation from 3D Reconstructed Droplets'', submitted at Indian Institute of Technology, Kanpur for the partial fulfilment of the requirements for the degree of Master of Technology. Towards the broad goal of reconstruction of a transparent object, this method presents a novel approach for 3D reconstruction of static liquid droplets on smooth, homogenous surfaces. The following studies were performed. First, a high resolution mobile camera equipped with macro lens is used for image acquisition of small-sized droplet. Next, for estimating intrinsic and extrinsic camera parameters a printed pattern was used. After that, U-Net CNN architecture was used to extract silhouettes of droplets from digital images using semantic segmentation. Finally, shape-from-silhouette method was employed with space carving algorithm to estimate the visual hull containing the droplet volume. The following is the flowdiagram of the methodology.

<p align="center">
  <img src="https://github.com/rawakash66/Thesis_Akash_2022/blob/main/Carving%20methodology.png" width="600">
</p>

## Installation <a name="installation"></a>
### Requirements
1. python3
2. numpy
3. matplotlib
4. opencv-gpu
5. tensorflow-gpu
6. segmentation-models
7. albumentations

Python can be installed using <a href="https://www.anaconda.com/" target="_blank">anacoda</a>. For installing OpenCV with GPU support follow the instructions provided in this <a href="https://www.youtube.com/watch?v=HsuKxjQhFU0" target="_blank">video</a>. All other libraries can be installed after this by executing the below code.

```python
pip install -U -r requirements.txt
```

## Camera calibration <a name="calibration"></a>
The camera is calibrated using 80 images of asymmetrical circle [pattern](https://github.com/rawakash66/Thesis_Akash_2022/blob/main/pattern%20circles.png). The pattern was printed on a photo paper of size 10 X 7.3 mm. The dataset acquired can be found at this [link](https://github.com/rawakash66/Thesis_Akash_2022/tree/main/camera%20calibration/data). To get the results on calibration parameters, run the script file provided in the [link](https://github.com/rawakash66/Thesis_Akash_2022/tree/main/camera%20calibration/script). The output will be the intrinsic parameters along with plots for lens distortion and reprojection errors as shown below.

<p align = "center">
  <img src = "https://github.com/rawakash66/Thesis_Akash_2022/blob/main/circle%20reprojection%20error.png" width = "400" />
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src = "https://github.com/rawakash66/Thesis_Akash_2022/blob/main/lens%20distortion.png" width = "400" />
</p>
