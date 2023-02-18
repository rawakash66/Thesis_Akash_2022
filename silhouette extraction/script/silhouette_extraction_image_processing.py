import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def filter_image(image, mask):
    r = image[:,:] * mask
    return r


if __name__=='__main__':
    # load image
    image_path = "../data/image_and_masks/train_imgs/train/IMG_20220226_152325568.png"
    img = cv2.imread(image_path)
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original image')
    plt.axis('off')

    # smoothing image
    imgx = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgx = cv2.GaussianBlur(imgx, (5, 5), 0)

    # K-Means
    ## preprocess
    twoDimg = imgx.reshape((-1, 3))
    twoDimg = np.float32(twoDimg)
    ## param definition
    K = 2
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ## apply K-means
    ret, label, center = cv2.kmeans(twoDimg, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((imgx.shape))
    fig = plt.figure()
    plt.imshow(result_image, cmap = 'gray')
    plt.title('K-Means segmentation')
    plt.axis('off')

    # Countour detection
    ## preprocessing
    _, thresh = cv2.threshold(imgx, np.mean(imgx), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    ## detect and draw contour
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key = cv2.contourArea)[-1]
    mask = np.zeros(edges.shape, np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    fig = plt.figure()
    plt.imshow(masked, cmap = 'gray')
    plt.title('Contour detection algo')
    plt.axis('off')

    # Thresholding
    thresh = threshold_otsu(imgx)
    img_otsu  = imgx < thresh
    filtered = filter_image(imgx, img_otsu)
    fig = plt.figure()
    plt.imshow(filtered, cmap = 'gray')
    plt.title('Thresholding algo')
    plt.axis('off')

    # Watershed algo
    ## preprocess
    ret, thresh = cv2.threshold(imgx, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ## noise removal by morphological opening
    ### noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    ### sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    ### Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    ### Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ## get markers
    ### Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    ### Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    ### Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    fig = plt.figure()
    plt.imshow(markers)
    plt.title('Watershed algo')
    plt.axis('off')
