import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import segmentation_models as sm

sm.set_framework('tf.keras')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 600
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('axes', linewidth=2)


if __name__=='__main__':
    # global params
    backbone = 'efficientnetb4'
    shape = 224
    preprocess_input = sm.get_preprocessing(backbone)
    model_path = '../data/model/unet_model'
    img_path = '../data/image_and_masks/train_imgs/train/IMG_20220426_132232945.png'
    choosen_layer = 'decoder_stage4a_conv'
    W, H = (3264, 1836)

    # load model
    model = sm.Unet(backbone, input_shape = (shape, shape, 3), classes = 1, activation = 'sigmoid')
    model.load_weights(model_path).expect_partial()
    # print(model.summary())

    # load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (shape, shape), interpolation = cv2.INTER_NEAREST_EXACT)
    img_array = np.expand_dims(img, axis=0)

    # get features
    layer_output = model.get_layer(choosen_layer).output
    feature_map_model = tf.keras.models.Model(model.input, layer_output)
    feature_map = feature_map_model.predict(img_array)
    
    # create feature image
    k = feature_map.shape[-1]
    size=feature_map.shape[1]
    images = []
    for i in range(k):
        feature_image = feature_map[0, :, :, i]
        feature_image-= feature_image.mean()
        feature_image/= feature_image.std()
        feature_image*=  64
        feature_image+= 128
        feature_image= np.clip(feature_image, 0, 255).astype('uint8')
        feature_image = cv2.resize(feature_image, (W, H), interpolation = cv2.INTER_NEAREST_EXACT)
        images.append(feature_image)
    
    # plot feature map
    fig, axs = plt.subplots(4, 4, figsize=(10,6)) 
    for i in range(k):
        axs[i//4, i%4].imshow(images[i])
        axs[i//4, i%4].axis('off')
    plt.tight_layout()
    plt.savefig('feature_map.png')