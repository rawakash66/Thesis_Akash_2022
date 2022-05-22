# Computer Vision Based Liquid Contact Angle Estimation from 3D Reconstructed Droplets

## Table of contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Camera calibration](#calibration)
* [Silhouette extraction](#silhouette)

## Introduction <a name="introduction"></a>
This repository includes the data and code scripts utilized in the thesis titled "Computer Vision Based Liquid Contact Angle Estimation from 3D Reconstructed Droplets", submitted at Indian Institute of Technology, Kanpur for the partial fulfilment of the requirements for the degree of Master of Technology. 
Towards the broad goal of reconstruction of a transparent object, this method presents a novel approach for 3D reconstruction of static liquid droplets on smooth, homogenous surfaces. The following studies were performed. 
First, a high resolution mobile camera equipped with macro lens is used for image acquisition of small-sized droplet. 
Next, for estimating intrinsic and extrinsic camera parameters a printed pattern was used. 
After that, U-Net CNN architecture was used to extract silhouettes of droplets from digital images using semantic segmentation. 
Finally, shape-from-silhouette method was employed with space carving algorithm to estimate the visual hull containing the droplet volume. 
The following is the flowdiagram of the methodology.

<p align="center">
  <img src="https://github.com/rawakash66/Thesis_Akash_2022/blob/main/figures/reconstruction%20methodology.png" width="600">
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

Python can be installed using <a href="https://www.anaconda.com/" target="_blank">anacoda</a>. For installing OpenCV with GPU support follow the instructions provided in this <a href="https://www.youtube.com/watch?v=HsuKxjQhFU0" target="_blank">video</a>. 

### Setup
For using this repository, run the following commands in your conda prompt window. <br/>

1. Install git using conda.
```python
conda install git
```
2. Clone the repository in your system.
```python
git clone https://github.com/rawakash66/Thesis_Akash_2022
```
3. Move inside the cloned folder.
```python
cd Thesis_Akash_2022
```
4. All other libraries mentioned can be installed after this by executing the below code.
```python
pip install -U -r requirements.txt
```
5. To run a script named "example.py" use below command and provide path location of the python file relative to the root cloned folder. 
```python
python PATH_TO_DIR\example.py
```

## Camera calibration <a name="calibration"></a>
The camera is calibrated using 80 images of asymmetrical circle [pattern](https://github.com/rawakash66/Thesis_Akash_2022/blob/main/figures/pattern%20circles.png). The pattern was printed on a photo paper of size 10 X 7.3 mm. The dataset acquired can be found at this [link](https://github.com/rawakash66/Thesis_Akash_2022/tree/main/camera%20calibration/data). To get the results on calibration parameters, run the script file provided in the [link](https://github.com/rawakash66/Thesis_Akash_2022/blob/main/camera%20calibration/script/calibration.py). The output will be the intrinsic parameters along with plots for reprojection errors and lens distortion as shown below.

<p align = "center">
  <img src = "https://github.com/rawakash66/Thesis_Akash_2022/blob/main/figures/reprojection%20error.png" width = "500" />
  <img src = "https://github.com/rawakash66/Thesis_Akash_2022/blob/main/figures/lens%20distortion.png" width = "500" />
</p>

## Silhouette extraction <a name="silhouette"></a>
The segmentation of droplets in the image is required before starting the reconstruction. 
An [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) CNN architecture was used because of its great performance in medical image segmentation of irregular-shaped cells. 
The [EfficientNetB4](https://arxiv.org/abs/1905.11946) was used as the backbone of the model and the pre-trained weights of imagenet was used in the encoder layer. 
The [segmentation-model](https://github.com/qubvel/segmentation_models) library was utilized for all the purposes. The dataset included 373 images for training, 40 images for validation and 24 images for testing. 
The image dataset included a combination of all the specimen used in the experiment along with some unseen examples to generalize the model. 
The dataset can be found at this [link](https://github.com/rawakash66/Thesis_Akash_2022/tree/main/silhouette%20extraction/data/images_and_masks). 
A helper jupyter notebook is provided to create ground truths for new images in the following [link](https://github.com/rawakash66/Thesis_Akash_2022/blob/main/silhouette%20extraction/notebook/labelling%20notebook.ipynb) and the utility code can be found [here](https://github.com/rawakash66/Thesis_Akash_2022/blob/main/silhouette%20extraction/script/Labelling.py).
To use this jupyter notebook run the following code in conda prompt to use matplotlib widget for interactive plot.

1. Install ipympl for interactive plot.
```python
pip install ipympl
```
2. Install nodejs and jupyterlab extension to use ipympl in jupyter notebook.
```python
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
```
To use the package restart the conda prompt after running these codes.
