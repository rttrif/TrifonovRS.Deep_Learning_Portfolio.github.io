"""
PROJECT 11: Underwater Segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: Autoencoder for semantic segmentation
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training Autoencoder
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

# %%
# PATH & LABEL PROCESS

# Main path
image_data_path = Path('data/train_val/images')
mask_data_path = Path('data/train_val/masks')

# Listing files
image_path = list(image_data_path.glob(r"*.jpg"))
mask_path = list(mask_data_path.glob(r"*.bmp"))

# Checking results
print(len(image_path))
print(len(mask_path))

image_path = sorted(image_path)
mask_path = sorted(mask_path)

# Transformation to series
image_series = pd.Series(image_path, name="image", dtype='object').astype(str)
mask_series = pd.Series(mask_path, name="mask", dtype='object').astype(str)

# Checking results
print(image_series)
print(mask_series)

# Concatenating series to train_data dataframe
train_df = pd.concat([image_series, mask_series], axis=1)

# Checking results
print(train_df.head())
print(train_df.info())

# %%
# VISUALIZATION: Overview of images

# Random image and mask example
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask.shape)
axis[1].set_title("Mask")
axis[1].imshow(example_mask)

plt.tight_layout()
plt.show()

# %%
# TRANSFORMATION LAYER
# Random example
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

copy_image = example_image.copy()
copy_image[example_mask == 0] = 0

figure, axis = plt.subplots(1, 3, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask.shape)
axis[1].set_title("Mask")
axis[1].imshow(example_mask)

axis[2].set_xlabel(copy_image.shape)
axis[2].set_title("Mask Transformation")
axis[2].imshow(copy_image)

plt.tight_layout()
plt.show()

# %%
# 2D-MASK
# Random example
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask[:, :, 0].shape)
axis[1].set_title("Mask")
axis[1].imshow(example_mask[:, :, 0] == 255)

plt.tight_layout()
plt.show()

# %%
# COMPARISON OF DIFFERENT DISPLAY OPTIONS
# Random example
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

object_path_trans = example_mask[:, :, 0] == 0

copy_image = example_image.copy()
copy_image[object_path_trans] = [255, 0, 0]

figure, axis = plt.subplots(1, 4, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(example_mask[:, :, 0].shape)
axis[1].set_title("2D-Mask")
axis[1].imshow(example_mask[:, :, 0] == 255)

axis[2].set_xlabel(copy_image.shape)
axis[2].set_title("Mask Transformation")
axis[2].imshow(copy_image)

axis[3].set_xlabel(example_mask.shape)
axis[3].set_title("Original mask")
axis[3].imshow(example_mask)

plt.tight_layout()
plt.show()

# %%
# COLOR SPACE

# 3-D SPACE
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)

Red_I, Green_I, Blue_I = cv2.split(example_image)

figure = plt.figure(figsize=(7, 7))
axis_func = figure.add_subplot(1, 1, 1, projection="3d")

Pixel_Colors_I = example_image.reshape((np.shape(example_image)[0] * np.shape(example_image)[1], 3))
Normalize_I = colors.Normalize(vmin=-1., vmax=1.)
Normalize_I.autoscale(Pixel_Colors_I)
Result_Pixel = Normalize_I(Pixel_Colors_I).tolist()

axis_func.scatter(Red_I.flatten(), Green_I.flatten(), Blue_I.flatten(), facecolors=Result_Pixel, marker=".")

axis_func.set_xlabel("Red")
axis_func.set_ylabel("Green")
axis_func.set_zlabel("Blue")

plt.tight_layout()
plt.show()

# %%
# SUB LAYER
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

Red_I, Green_I, Blue_I = cv2.split(example_image)

copy_image = example_image.copy()
image_path_trans = copy_image[:, :, 0] - Blue_I

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(image_path_trans.shape)
axis[1].set_title("Image path trans")
axis[1].imshow(image_path_trans)

plt.tight_layout()
plt.show()

ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

Red_I, Green_I, Blue_I = cv2.split(example_image)

copy_image = example_image.copy()
image_path_trans = copy_image[:, :, 0] - Green_I

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(image_path_trans.shape)
axis[1].set_title("Image path trans")
axis[1].imshow(image_path_trans)

plt.tight_layout()
plt.show()

# %%
# COMPARISON OF DIFFERENT DISPLAY OPTIONS
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

Red_I, Green_I, Blue_I = cv2.split(example_image)

copy_image = example_image.copy()
image_path_trans = copy_image[:, :, 0] - (Red_I + Blue_I)

figure, axis = plt.subplots(1, 3, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(image_path_trans.shape)
axis[1].set_title("Image path trans")
axis[1].imshow(image_path_trans)

axis[2].set_xlabel(example_mask.shape)
axis[2].set_title("Original mask")
axis[2].imshow(example_mask)

plt.tight_layout()
plt.show()


ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

Red_I, Green_I, Blue_I = cv2.split(example_image)

copy_image = example_image.copy()
image_path_trans = copy_image[:, :, 0] - (Red_I + Green_I + Blue_I)

figure, axis = plt.subplots(1, 3, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(image_path_trans.shape)
axis[1].set_title("Image path trans")
axis[1].imshow(image_path_trans)

axis[2].set_xlabel(example_mask.shape)
axis[2].set_title("Original mask")
axis[2].imshow(example_mask)

plt.tight_layout()
plt.show()

# %%
# SUB LAYER MASK
ind = np.random.randint(0, train_df["image"].shape[0])

example_image = cv2.cvtColor(cv2.imread(train_df["image"][ind]), cv2.COLOR_BGR2RGB)
example_mask = cv2.cvtColor(cv2.imread(train_df["mask"][ind]), cv2.COLOR_BGR2RGB)

Red_I, Green_I, Blue_I = cv2.split(example_image)

copy_image = example_image.copy()
image_path_trans = copy_image[:,:,0] == Green_I

figure, axis = plt.subplots(1, 3, figsize=(15, 15))

axis[0].set_xlabel(example_image.shape)
axis[0].set_title("Image")
axis[0].imshow(example_image)

axis[1].set_xlabel(image_path_trans.shape)
axis[1].set_title("Image path trans")
axis[1].imshow(image_path_trans)

axis[2].set_xlabel(example_mask.shape)
axis[2].set_title("Original mask")
axis[2].imshow(example_mask)

plt.tight_layout()
plt.show()

# SUB COLOR SPACE
