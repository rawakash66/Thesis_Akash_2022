# Computer vision-based on-site estimation of contact angle from 3D reconstruction of droplets

## Table of contents
* [Introduction](#introduction)
* [Installation](#installation)
  1. [Requirements](#requirements)
  2. [Setup](#setup) 
* [Camera calibration](#calibration)
* [Silhouette extraction](#silhouette)
* [Droplet reconstruction](#reconstruction)
* [References](#references)

## Introduction <a name="introduction"></a>
Current methods to measure the contact angle require orthogonal imaging of the droplet and substrate. We have developed a novel computer vision-based technique to reconstruct the surface of the 3D transparent microdroplet from non-orthogonal images and determined  the contact angle using custom-made equipment comprising a smartphone camera and macro lens. After estimating the  intrinsic and extrinsic camera parameters using a printed pattern, the EfficientNet-B4 model of U-Net CNN architecture was used to extract silhouettes of droplets from images using semantic segmentation. Finally, the shape-from-silhouette method was employed involving a space carving algorithm to estimate the visual hull containing the droplet volume. Comparison with measurements from a state-of-the-art goniometer of static and dynamic contact angles on various substrates using a standard goniometer revealed an average error of 4%.  Our method, using non-orthogonal images, was found to be successful for the on-site measurement of static and dynamic contact angles, as well as 3D reconstruction of the transparent droplets.

<p align="center">
  <img src="https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/methodology.png" width="800">
</p>

## Installation <a name="installation"></a>
### Requirements <a name="requirements"></a>
1. python3
2. numpy
3. matplotlib
4. opencv-gpu
5. tensorflow-gpu
6. segmentation-models
7. albumentations

Python can be installed using <a href="https://www.anaconda.com/" target="_blank">anacoda</a>. For installing OpenCV with GPU support follow the instructions provided in this <a href="https://www.youtube.com/watch?v=HsuKxjQhFU0" target="_blank">video</a>. 

### Setup <a name="setup"></a>
For using this repository, run the following commands in your conda prompt window. <br/>

1. Clone the repository in your system.
```python
git clone https://github.com/rawakash66/transparent-drop-reconstruction
```
2. Move inside the cloned folder.
```python
cd transparent-drop-reconstruction
```
3. All other libraries mentioned can be installed after this by executing the below code.
```python
pip install -U -r requirements.txt
```

## Camera calibration <a name="calibration"></a>
The camera is calibrated using 80 images of asymmetrical circle [pattern](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/pattern%20circles.png). The pattern was printed on a photo paper of size 10 X 7.3 mm. 
The dataset acquired can be found at this [link](https://github.com/rawakash66/transparent-drop-reconstruction/tree/main/camera%20calibration/data). 
To get the results on calibration parameters, run the script file provided in the [link](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/camera%20calibration/script/calibration.py). 
The output will be the intrinsic parameters including focal length, principal point, radial distortion coefficients and tangential distortion coefficients along with plots for [reprojection errors](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/reprojection%20error.png) and [lens distortion](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/lens%20distortion.png). 
A yml file also gets created inside the data folder containing the saved intrinsic camera matrix and distortion coefficients necessary for reconstruction process.
The file can be viewed [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/camera%20calibration/data/calibration_circle.yml).

## Silhouette extraction <a name="silhouette"></a>
The segmentation of droplets in the image is required before starting the reconstruction. 
An U-Net CNN architecture was used because of its state-of-art performance in medical image segmentation of irregular-shaped cells. 
The EfficientNet-B4 was used as the backbone of the model and the pre-trained weights of imagenet was used in the encoder layer. 
The segmentation-model library was utilized for all the purposes.
The image dataset included a combination of all the specimen used in the experiment along with some unseen examples to generalize the model. 
The dataset can be found at this [link](https://github.com/rawakash66/transparent-drop-reconstruction/tree/main/silhouette%20extraction/data/images_and_masks). 
A helper jupyter notebook is provided to create ground truths for new images in the following [link](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/silhouette%20extraction/notebook/labelling%20notebook.ipynb) and the utility code for the same can be found [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/silhouette%20extraction/script/Labelling.py).
To use this jupyter notebook, run the following code in conda prompt to use matplotlib widget for interactive plot.

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
The jupyter notebook can be opened in the jupyterlab. <br/>

The link to the script for training the model can be found [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/silhouette%20extraction/script/segmentation.py).
The parameters such as 'backbone', 'epoch', 'learning rate', 'batch size', etc. can be changed accordingly to train the model of your choice.
The output of the script include the plot for [learning curve](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/learning%20curve.png) and prediction on top 5 images of validation and test dataset.
One of the example prediction on validation data can be found [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/DL%20val%20pred%201.png) and prediction on unseen test data is provided [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/DL%20prediction.png). 
The script also provides the best model which gets automatically saved [here](https://github.com/rawakash66/transparent-drop-reconstruction/tree/main/silhouette%20extraction/script/model).
It includes training checkpoints and model weights which are necessary for inference during reconstruction process.
The image of the model architecture used can be found [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/unet.png).

## Droplet reconstruction <a name="reconstruction"></a>
Once the camera calibration and model training is completed, copy and paste the yml file created from calibration step inside the 'droplet reconstruction/data/intrinsic' folder [here](https://github.com/rawakash66/transparent-drop-reconstruction/tree/main/droplet%20reconstruction/data/intrinsics).
Similarly, copy and paste the model folder from reconstruction step inside the 'droplet reconstruction/data/model' folder [here](https://github.com/rawakash66/transparent-drop-reconstruction/tree/main/droplet%20reconstruction/data/model).
**If you don't want to train your model, you can download my trained model weights from the link [here](https://drive.google.com/uc?id=1U_Tn5klWV0zW8UWyo0dj-kXuQw9FZ4hX&export=download).
Extract the zip file inside the folder path "transparent-drop-reconstruction/droplet reconstruction/data/" to use the weights.**
The python script to reconstruct the droplet can be found [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/droplet%20reconstruction/script/reconstruction.py).
After completing above steps, run this python script to get the point cloud representing the droplet structure.
You can change the thresholding values to tune the shape of the droplet.
The output of the script includes the 'shape.vtk' and 'shape.ply' file which can be found inside the [mesh](https://github.com/rawakash66/transparent-drop-reconstruction/tree/main/droplet%20reconstruction/data/mesh) folder.
The 'shape.ply' file is imported in the MeshLab for visualization and postprocessing.
The convex hull was used to get the enclosed mesh for the droplet and Z-painting was used to smooth out the surfaces.
Following is the image of the reconstructed and processed droplet for PLA specimen at 30 degree tilt angle. 

<p align="center">
  <img src="https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/figures/reconstructed%20droplet.png" width="500">
</p>

A numerical-based approach was also used to reconstruct the droplet. The MATLAB file for the same can be found [here](https://github.com/rawakash66/transparent-drop-reconstruction/blob/main/droplet%20reconstruction/script/numerical_reconstruction.m).

## References <a name="references"></a>
I declare that the code scripts used in this repository is from open source community and I do not claim any copyright on the same.
Following are the citations to the sources.


* [Anaconda](https://anaconda.org/)
* [MeshLab](https://www.meshlab.net/)
* [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
* [EfficientNets](https://arxiv.org/abs/1905.11946)
* [segmentation-model](https://github.com/qubvel/segmentation_models)
* [Camera calibration using circle grid](https://longervision.github.io/2017/03/18/ComputerVision/OpenCV/opencv-internal-calibration-circle-grid/)
* [Lens distortion plot](http://amroamroamro.github.io/mexopencv/opencv/calibration_demo.html)
* [Labelling](https://github.com/ianhi/AC295-final-project-JWI)
* [Segmentation](https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb)
* [Space carving](https://github.com/zinsmatt/SpaceCarving/blob/master/space_carving.py)
