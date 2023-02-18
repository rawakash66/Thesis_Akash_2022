import os
import shutil
from PIL import Image
from tqdm import tqdm


def jpeg_to_png(source, destination):
    # function to convert jpg or jpeg to png
    # source and destination are folder paths
    if not os.path.exists(destination):
        os.mkdir(destination)
    else:
        shutil.rmtree(destination)
        os.mkdir(destination)

    for img in tqdm(sorted(os.listdir(source))):
        if os.path.isdir(source + img):
            continue
        im1 = Image.open(source + img)
        if img[-4:] == 'jpeg':
            im1.save(destination + img[:-4] +'png')
        else:
            im1.save(destination + img[:-3] +'png')


def delete_uncommon_images(src, dst):
    # function to delete images from src folder
    # if not present in dst folder
    # src and dst are folder paths
    dstlist = os.listdir(dst)
    deleted_count = 0
    for file in tqdm(sorted(os.listdir(src))):
        if file not in dstlist:
            os.remove(src + file)
            deleted_count += 1
    print("Deleted files", deleted_count)


def resize_images(source, destination, height, width):
    # function to resize image from source folder
    # and save it to destination
    if not os.path.exists(destination):
        os.mkdir(destination)
    else:
        shutil.rmtree(destination)
        os.mkdir(destination)

    for img in tqdm(sorted(os.listdir(source))):
        if os.path.isdir(source + img):
            continue
        im1 = Image.open(source + img)
        im1 = im1.resize((width, height))
        im1.save(destination + img[:-3] +'jpg')


if __name__=='__main__':
    source = "C:/Users/uplcu/Downloads/Akash data/"
    destination = "C:/Users/uplcu/Downloads/images/"
    jpeg_to_png(source, destination)

    src = "C:/Users/uplcu/Thesis/Silhouette Extraction Files/data/image_and_masks/train_mask/train/"
    dst = "C:/Users/uplcu/Thesis/Silhouette Extraction Files/data/image_and_masks/train_imgs/train/"
    delete_uncommon_images(src, dst)

    source = "C:/Users/uplcu/Thesis/Camera Calibration Files/data/images_chessboard/"
    destination = "C:/Users/uplcu/Downloads/images/"
    H, W = (1499, 1932)
    resize_images(source, destination, H, W)
