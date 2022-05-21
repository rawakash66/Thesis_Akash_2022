#import libraries
import os
import cv2
from tensorflow import keras
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm

sm.set_framework('tf.keras')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 600
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('axes', linewidth=2)

# for visualizing images
def visualize(**images):
    '''
    Parameters
    ----------
    **images : object
        data generator object.

    Returns
    -------
    None.
    '''
    n = len(images)
    plt.figure(figsize=(15, 45))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        plt.axis('off')
    plt.show()
 
# helper function for data visualization    
def denormalize(img):
    '''
    Parameters
    ----------
    img : np.array
        unnormalized image.

    Returns
    -------
    img : np.array
        normalized image.
    '''
    img_max = np.percentile(img, 98)
    img_min = np.percentile(img, 2)    
    img = (img - img_min) / (img_max - img_min)
    img = img.clip(0, 1)
    return img

# helper function to round and clip values
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# class for dataset and preprocessing
class Dataset:
    
    CLASSES = ['drop']
    
    def __init__(self, images_dir, shape, masks_dir = None, classes = None, augmentation = None, preprocessing = None):
        '''
        Parameters
        ----------
        images_dir : str
            path to image directory.
        shape : int
            shape dimension of input
        masks_dir : str
            path to mask directory.
        classes : list, optional
            list of class names. The default is None.
        augmentation : obj, optional
            data transformation pipeline. The default is None.
        preprocessing : obj, optional
            data preprocessing pipeline. The default is None.

        Returns
        -------
        None.
        '''
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.shape = shape
        self.class_values = [self.CLASSES.index(cls.lower())+1 for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __getitem__(self, i):
        '''
        Parameters
        ----------
        i : int
            image index.

        Returns
        -------
        image and mask.
        '''
        # load data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST_EXACT)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST_EXACT)
        
        # extract classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis = -1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis = -1, keepdims = True)
            mask = np.concatenate((mask, background), axis = -1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask)
        
        return image, mask
    
    def __len__(self):
        return len(self.ids)

# class for data loader
class Dataloader(keras.utils.Sequence):
    def __init__(self, dataset, batch_size = 1, shuffle = False):
        '''
        Parameters
        ----------
        dataset : obj
            instance of Dataset class for image loading and preprocessing.
        batch_size : int, optional
            number of images in a batch. The default is 1.
        shuffle : bool, optional
            boolean for shuffling the image or not. The default is False.

        Returns
        -------
        None.
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()
    
    def __getitem__(self, i):
        '''
        Parameters
        ----------
        i : int
            image index.

        Returns
        -------
        batch of images.
        '''
        # collect batch data
        start = i*self.batch_size
        stop = (i + 1)*self.batch_size
        data = []
        
        for i in range(start, stop):
            data.append(self.dataset[i])
            
        # transpose list of lists
        batch = [np.stack(samples, axis = 0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        return len(self.indexes)//self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# function for augumentation
def get_training_augmentation():
    '''
    Returns
    -------
    training augmentation pipeline.
    '''
    train_transform = [
        A.HorizontalFlip(p = 0.5),
        A.GaussNoise(p = 0.2),
        
        A.OneOf(
            [
                A.CLAHE(p = 1),
                A.RandomBrightness(p = 1),
                A.RandomGamma(p = 1),
                ], p = 0.9),
        A.OneOf(
            [
                A.Sharpen(p = 1),
                A.Blur(blur_limit = 3, p = 1),
                A.MotionBlur(blur_limit = 3, p = 1),
                ], p = 0.9),
        A.OneOf(
            [
                A.RandomContrast(p = 1),
                A.HueSaturationValue(p = 1),
                ], p = 0.9),
        A.Lambda(mask = round_clip_0_1)
        ]
    
    return A.Compose(train_transform)

# function for image preprocessing
def get_preprocessing(preprocessing_fn):
    '''
    Parameters
    ----------
    preprocessing_fn : obj
        function for preprocessing.

    Returns
    -------
    preprocessing pipeline.
    '''
    _transform = [
        A.Lambda(image = preprocessing_fn),
    ]
    
    return A.Compose(_transform)

# function to get model
def get_model(backbone, classes, LR, shape):
    '''
    Parameters
    ----------
    backbone : str
        backbone for the encoder layer.
    classes : list
        list of class strings.
    shape : int
        dimension of input image
    LR : float
        learning rate

    Returns
    -------
    compiled U-Net model
    '''
    # parameter initialization
    n_classes = 1 if len(classes) == 1 else (len(classes) + 1)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    
    # model 
    model = sm.Unet(backbone_name = backbone, input_shape = (shape, shape, 3), classes = n_classes, activation = activation)
    
    # optimizer
    opt = keras.optimizers.Adam(LR)
    
    # losses
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    # metrices
    metrics = [sm.metrics.IOUScore(threshold = 0.5), sm.metrics.FScore(threshold = 0.5)]
    
    # compiling
    model.compile(opt, total_loss, metrics)
    
    return model

# function for training
def train(model, train_gen, val_gen, n_epochs, batch_size):
    '''
    Parameters
    ----------
    model : obj
        U-Net compiled model.
    train_gen : obj
        train data generator.
    val_gen : obj
        val data generator.
    n_epochs : int
        number of epochs to train.
    batch_size : int
        size of each batch

    Returns
    -------
    training history.
    '''
    # callbacks
    callbacks = [keras.callbacks.ModelCheckpoint('model/unet_model', save_weights_only=True, save_best_only=True, mode='min'), keras.callbacks.ReduceLROnPlateau()]
    
    # training
    history = model.fit_generator(train_gen,
                                  steps_per_epoch = len(train_gen),
                                  epochs = n_epochs,
                                  callbacks = callbacks,
                                  validation_data = val_gen,
                                  validation_steps = len(val_gen))
    
    return history

# function for plotting learning curves
def plot_learning_curves(history, n_epochs):
    '''
    Parameters
    ----------
    history : dict
        model training history.
    n_epochs : int
        number of epochs to train

    Returns
    -------
    None.
    '''
    fig, ax = plt.subplots(1, 3, figsize = (40, 12))
    epochs = range(n_epochs)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    ax[0].plot(epochs, loss, 'r', label = 'Training loss', linewidth = 3)
    ax[0].plot(epochs, val_loss, 'b', label = 'Validation loss', linewidth = 3)
    ax[0].set_ylabel('Loss', fontsize = 40)
    
    f1_score = history.history['f1-score']
    val_f1_score = history.history['val_f1-score']
    ax[1].plot(epochs, f1_score, 'r', label = 'Training F1 score', linewidth = 3)
    ax[1].plot(epochs, val_f1_score, 'b', label = 'Validation F1 score', linewidth = 3)
    ax[1].set_ylabel('F1 score', fontsize = 40)
    ax[1].set_ylim(0, 1)
    
    iou_score = history.history['iou_score']
    val_iou_score = history.history['val_iou_score']
    ax[2].plot(epochs, iou_score, 'r', label = 'Training IOU score', linewidth = 3)
    ax[2].plot(epochs, val_iou_score, 'b', label = 'Validation IOU score', linewidth = 3)
    ax[2].set_ylabel('IOU', fontsize = 40)
    ax[2].set_ylim(0, 1)
    
    for i in range(3):
        ax[i].set_xlabel('Epochs', fontsize = 40)
        ax[i].legend(fontsize = 38)
        ax[i].tick_params(axis='both', which='major', labelsize=32)
    
    plt.show()
    
if __name__ == "__main__":
    # path to data
    DATA_DIR = '../data/image_and_masks/'
    x_train_dir = os.path.join(DATA_DIR, 'train_imgs/train')
    y_train_dir = os.path.join(DATA_DIR, 'train_mask/train')
    x_val_dir = os.path.join(DATA_DIR, 'val_imgs/val')
    y_val_dir = os.path.join(DATA_DIR, 'val_mask/val')
    x_test_dir = "../data/image_and_masks/test_imgs/test/"
    
    # global parameters
    backbone = 'efficientnetb4'
    classes = ['drop']
    batch_size = 16
    n_epochs = 40
    LR = 0.0001
    shape = 224
    preprocess_input = sm.get_preprocessing(backbone)
    save_path = 'model/unet_model'
    
    # data handling
    train_dataset = Dataset(x_train_dir,
                            shape,
                            y_train_dir,
                            classes = classes,
                            augmentation = get_training_augmentation(),
                            preprocessing = get_preprocessing(preprocess_input))
    
    val_dataset = Dataset(x_val_dir,
                          shape,
                          y_val_dir,
                          classes = classes,
                          preprocessing = get_preprocessing(preprocess_input))
    # data generators
    train_gen = Dataloader(train_dataset, batch_size = batch_size, shuffle = True)
    val_gen = Dataloader(val_dataset, batch_size = 1, shuffle = False)
    
    # get model and training
    model = get_model(backbone, classes, LR, shape)
    history = train(model, train_gen, val_gen, n_epochs, batch_size)
    
    # plot learning curves
    plot_learning_curves(history, n_epochs)
    
    # evaluation of validation data
    model.load_weights(save_path)
    scores = model.evaluate_generator(val_gen)
    print("Loss: {:.5}".format(scores[0]))
    print("Mean IOU score: {:.5}".format(scores[1]))
    print("Mean f1 score: {:.5}".format(scores[2]))
    
    # prediction on validation images
    ids = np.arange(len(val_dataset))
    
    for i in ids[:5]:
        image, gt_mask = val_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()
        image = cv2.resize(image.squeeze(), (3264, 1836), interpolation = cv2.INTER_NEAREST_EXACT)
        gt_mask = cv2.resize(gt_mask[..., 0].squeeze(), (3264, 1836), interpolation = cv2.INTER_NEAREST_EXACT)
        pr_mask = cv2.resize(pr_mask[..., 0].squeeze(), (3264, 1836), interpolation = cv2.INTER_NEAREST_EXACT)
        visualize(image = image, gt_mask = gt_mask, pr_mask = pr_mask)
        
    # prediction on test data
    test_ids = sorted(os.listdir(x_test_dir))
    test_imgs = [os.path.join(x_test_dir, image_id) for image_id in test_ids]
    
    for path in test_imgs[:5]:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (shape, shape), interpolation = cv2.INTER_NEAREST_EXACT)
        img = np.expand_dims(img, axis=0)
        
        pr_mask = model.predict(img).round()
        pr_mask = cv2.resize(pr_mask[..., 0].squeeze(), (3264, 1836), interpolation = cv2.INTER_NEAREST_EXACT)
        image = cv2.resize(img.squeeze(), (3264, 1836), interpolation = cv2.INTER_NEAREST_EXACT)
        visualize(image = image, pr_mask = pr_mask)