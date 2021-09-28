"""
PROJECT 12: Cityscapes Segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: Fully Convolutional Network(FCN)
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training Training Fully Convolutional Network
2.1 FCN 8
2.2 FCN 16
2.3 FCN 32

EXPLORATORY DATA ANALYSIS
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
from glob import glob
from tqdm import tqdm
import itertools

# %%
# PATH & LABEL PROCESS

# Main path
trian_data_path = 'data/cityscapes_data/train'
test_data_path = 'data/cityscapes_data/val'

train_images = []
train_masks = []
test_images = []
test_masks = []

def load_images(path):
    temp_img, temp_mask = [], []
    images = glob(os.path.join(path, '*.jpg'))
    for i in tqdm(images):
        i = cv2.imread(i)
        i = cv2.normalize(i, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        img = i[:, :256]
        msk = i[:, 256:]
        temp_img.append(img)
        temp_mask.append(msk)
    return temp_img, temp_mask


train_images, train_masks = load_images(trian_data_path)
test_images, test_masks = load_images(test_data_path)

# Checking results
print('TRAINING DATA')
print('Shape of training images: ', np.shape(train_images))
print('Shape of training masks: ', np.shape(train_masks))

print('TEST DATA')
print('Shape of test images: ', np.shape(test_images))
print('Shape of test masks: ', np.shape(test_masks))
# %%
# VISUALIZATION: Overview of images

# Random image and mask example
ind = np.random.randint(0, len(train_images))

example_image = train_images[ind]
example_mask = train_masks[ind]

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask.shape)
axis[1].set_title("Mask")
axis[1].imshow(example_mask)

plt.tight_layout()
plt.show()

# Random example masked image
ind = np.random.randint(0, len(train_images))

example_image = train_images[ind]
example_mask = train_masks[ind]

masked_image = example_image * 0.5 + example_mask * 0.5

figure, axis = plt.subplots(1, 3, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask.shape)
axis[1].set_title("Mask")
axis[1].imshow(example_mask)

axis[2].set_xlabel(masked_image.shape)
axis[2].set_title("Masked Image")
axis[2].imshow(masked_image)


plt.tight_layout()
plt.show()
