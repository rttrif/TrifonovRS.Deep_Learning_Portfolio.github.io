"""
PROJECT 15: Car segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: Mask R-CNN
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training Mask R-CNN
"""
# %%
# IMPORT LIBRARIES

# GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

# PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, \
    BatchNormalization, Permute, TimeDistributed, Bidirectional, GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, \
    SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D, Conv2DTranspose, ReLU, \
    UpSampling2D, Concatenate, Conv2DTranspose, Input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# %%
print(tf.config.list_physical_devices('GPU'))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# %%
# DATA PREPARATION
class Dataset(tf.Module):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_pth = os.path.join(self.img_dir, self.images[index])
        mask_pth = os.path.join(self.mask_dir, self.masks[index])

        image = cv2.imread(img_pth)
        image = image / 255

        mask = cv2.imread(mask_pth)
        mask = mask / 255

        image = np.array(image)
        mask = np.array(mask)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        if self.transform is not None:
            augment = self.transform(image=image, mask=mask)
            image = augment['image']
            mask = augment['mask']

        return image, mask


def get_loaders(train_dir, train_maskdir,
                train_transform=None, num_workers=4, pin_memory=True):
    train_ds = Dataset(
        train_dir,
        train_maskdir
    )

    train_img = []
    train_mask = []

    for im, ma in train_ds:
        train_img.append(im)
        train_mask.append(ma)

    train_img = np.array(train_img)
    train_mask = np.array(train_mask)

    return train_img, train_mask


# %%
# Main path
image_data_path = 'data/images'
mask_data_path = 'data/masks'

train_image, train_mask = get_loaders(image_data_path, mask_data_path)

# Checking results
print(f'Image shape: {np.shape(np.array(train_image))}')
print(f'Mask shape: {np.shape(np.array(train_mask))}')


# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

def learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(20, 8))
    plt.grid(True)
    plt.title('Learning curves')
    plt.gca().set_ylim(0, 1)
    plt.show()


# %%
# MASK R-CNN
def backboneNN():
    input_ = tf.keras.layers.Input((256, 256, 3))

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # 256

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # 128

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # 64

    x = tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    # 32

    featuremap = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='featuremap')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(featuremap)
    # 16

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(2, activation='sigmoid')(
        x)  # in case of multiclass one-hot encoding we need a sigmoid at the end
    featuremapmodel = tf.keras.Model(input_, featuremap, name="CNN_fm")
    classifiermodel = tf.keras.Model(input_, output, name="CNN")

    featuremapmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            loss=tf.keras.losses.BinaryCrossentropy())
    classifiermodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5))

    return classifiermodel, featuremapmodel


def rpnN(featuremap):
    # RPN modell

    initializer = tf.keras.initializers.GlorotNormal(seed=None)
    input_ = tf.keras.layers.Input(shape=[None, None, featuremap.shape[-1]], name="rpn_INPUT")

    shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', strides=1, name='rpn_conv_shared',
                                    kernel_initializer=initializer)(input_)
    x = tf.keras.layers.Conv2D(5 * 2, (1, 1), padding='valid', activation='linear', name='rpn_class_raw',
                               kernel_initializer=initializer)(shared)

    rpn_class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    rpn_probs = tf.keras.layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)  # --> BG/FG

    # Bounding box refinement. [batch, H, W, depth]
    x = tf.keras.layers.Conv2D(5 * 4, (1, 1), padding="valid", activation='linear', name='rpn_bbox_pred',
                               kernel_initializer=initializer)(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
    rpnN = tf.keras.models.Model(input_, outputs, name="RPN")

    return rpnN


def classheadNN(featurefilters, proposalcount, roisize):
    input_ = tf.keras.layers.Input((proposalcount, roisize[0], roisize[1], featurefilters))

    x = tf.keras.layers.Conv2D(kernel_size=(1, 1), padding='valid', activation='relu', filters=featurefilters)(input_)
    x = tf.debugging.check_numerics(x, 'x has nan')
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(kernel_size=(1, 1), padding='valid', activation='relu', filters=featurefilters)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    flatten = tf.keras.layers.Flatten()(x)

    beforeclass = tf.keras.layers.Dense(proposalcount * 3, name='beforeclasspred')(
        flatten)  # 3: numofclasses + 1 for background
    beforeclass = tf.debugging.check_numerics(beforeclass, 'beforeclass has nan')

    class_logits = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], proposalcount, 3]),
                                          name='classpred')(beforeclass)
    class_probs = tf.keras.layers.Activation("softmax", name="classhead_class")(class_logits)  # --> BG/FG

    beforebox = tf.keras.layers.Dense(3 * 4 * proposalcount, activation='linear', name='beforeboxpred')(
        flatten)  # 3 is the num of classes + 1 for background
    bboxpred = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], proposalcount, 3, 4]),
                                      name='boxrefinement')(
        beforebox)  # for every roi for every class we predict dx,dy,dw,dh
    outputs = [class_logits, class_probs, bboxpred]
    classheadNN = tf.keras.Model(input_, outputs, name="classhead")

    return classheadNN


def maskheadNN(featurefilters, proposalcount, maskroisize):
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    input_ = tf.keras.layers.Input((proposalcount, maskroisize[0], maskroisize[1], featurefilters))

    x = tf.keras.layers.Conv2D(kernel_size=(1, 1), padding='same', activation='relu', filters=featurefilters / 2)(
        input_)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(kernel_size=(3, 3), padding='same', activation='relu', filters=featurefilters / 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(kernel_size=(3, 3), padding='same', activation='relu', filters=featurefilters / 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(2, 2)))(x)
    x = tf.keras.layers.ReLU()(x)
    pred_mask = tf.keras.layers.Conv2D(kernel_size=(1, 1), padding='same', activation='sigmoid', filters=2)(
        x)  # 2 filters, as we predict a mask for each class

    maskheadNN = tf.keras.Model(input_, pred_mask, name="maskhead")

    return maskheadNN
