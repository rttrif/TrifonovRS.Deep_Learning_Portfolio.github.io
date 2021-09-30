"""
PROJECT 13:  People Clothing Segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: U-Net
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training U-Net

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

import tensorflow as tf
# %%
# PATH & LABEL PROCESS

# Main path
image_data_path = 'data/png_images'
mask_data_path = 'data/png_masks'

# A list to collect image paths
image_path = []
for root, dirs, files in os.walk(image_data_path):
    # iterate over 1000 images
    for file in files:
        # create path
        path = os.path.join(root,file)
        # add path to list
        image_path.append(path)
len(image_path)

# A list to collect masks paths
mask_path = []
for root, dirs, files in os.walk(mask_data_path):
    # iterate over 1000 masks
    for file in files:
        # obtain the path
        path = os.path.join(root,file)
        # add path to the list
        mask_path.append(path)
len(mask_path)

# Checking results
print(len(image_path))
print(len(mask_path))

image_path = sorted(image_path)
mask_path = sorted(mask_path)

# %%
# Read and decode the images and masks
images = []
# iterate over 1000 image paths
for path in tqdm(image_path):
    # read file
    file = tf.io.read_file(path)
    # decode png file into a tensor
    image = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
    # append to the list
    images.append(image)

# create a list to store masks
masks = []
# iterate over 1000 mask paths
for path in tqdm(mask_path):
    # read the file
    file = tf.io.read_file(path)
    # decode png file into a tensor
    mask = tf.image.decode_png(file, channels=1, dtype=tf.uint8)
    # append mask to the list
    masks.append(mask)

# Checking results
print(len(images))
print(len(masks))


# %%
# VISUALIZATION: Overview of images

# Random image and mask example
ind = np.random.randint(0, len(images))

example_image = images[ind]
example_mask = masks[ind]

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask.shape)
axis[1].set_title("Mask")
axis[1].imshow(example_mask)

plt.tight_layout()
plt.show()








