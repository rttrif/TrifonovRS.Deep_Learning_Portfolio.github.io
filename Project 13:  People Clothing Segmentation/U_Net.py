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
import matplotlib as mpl
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
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, \
    BatchNormalization, Permute, TimeDistributed, Bidirectional, GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, \
    SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D, Conv2DTranspose, ReLU, \
    UpSampling2D, concatenate, Conv2DTranspose, Input, Add, Activation

from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

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
        path = os.path.join(root, file)
        # add path to list
        image_path.append(path)
len(image_path)

# A list to collect masks paths
mask_path = []
for root, dirs, files in os.walk(mask_data_path):
    # iterate over 1000 masks
    for file in files:
        # obtain the path
        path = os.path.join(root, file)
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

print(f'Array image shape: {np.shape(np.array(images))}')
print(f'Array mask shape: {np.shape(np.array(masks))}')


# %%
# DATA PREPARATION

# Resize data
def resize_image(image):
    # scale the image
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    # resize image
    image = tf.image.resize(image, (256, 256))
    return image


def resize_mask(mask):
    # resize the mask
    mask = tf.image.resize(mask, (256, 256))
    mask = tf.cast(mask, tf.uint8)
    return mask


X = [resize_image(i) for i in images]
y = [resize_mask(m) for m in masks]

# Checking results
print(f'Length of X: {len(X)}')
print(f'Length of y: {len(y)}')

# Split Data
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Create tensorflow Dataset objects
train_X = tf.data.Dataset.from_tensor_slices(train_X)
val_X = tf.data.Dataset.from_tensor_slices(val_X)

train_y = tf.data.Dataset.from_tensor_slices(train_y)
val_y = tf.data.Dataset.from_tensor_slices(val_y)

# # Checking results:  the shapes and data types
train_X.element_spec, train_y.element_spec, val_X.element_spec, val_y.element_spec


# %%
# DATA AUGMENTATION

# Define Functions for data augmentation.
def brightness(img, mask):
    # adjust brightness of image
    # don't alter in mask
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask


def gamma(img, mask):
    # adjust gamma of image
    # don't alter in mask
    img = tf.image.adjust_gamma(img, 0.1)
    return img, mask


def hue(img, mask):
    # adjust hue of image
    # don't alter in mask
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask


def crop(img, mask):
    # crop both image and mask identically
    img = tf.image.central_crop(img, 0.7)
    # resize after cropping
    img = tf.image.resize(img, (128, 128))
    mask = tf.image.central_crop(mask, 0.7)
    # resize afer cropping
    mask = tf.image.resize(mask, (128, 128))
    # cast to integers as they are class numbers
    mask = tf.cast(mask, tf.uint8)
    return img, mask


def flip_hori(img, mask):
    # flip both image and mask identically
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask


def flip_vert(img, mask):
    # flip both image and mask identically
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask


def rotate(img, mask):
    # rotate both image and mask identically
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask


# %%
# Perform data augmentation with original training set and concatenate with enlarged training set.
# Zip images and masks
train = tf.data.Dataset.zip((train_X, train_y))
val = tf.data.Dataset.zip((val_X, val_y))

# perform augmentation on train data only

a = train.map(brightness)
b = train.map(gamma)
c = train.map(hue)
d = train.map(crop)
e = train.map(flip_hori)
f = train.map(flip_vert)
g = train.map(rotate)

# concatenate every new augmented sets
train = train.concatenate(a)
train = train.concatenate(b)
train = train.concatenate(c)
train = train.concatenate(d)
train = train.concatenate(e)
train = train.concatenate(f)
train = train.concatenate(g)


# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

def learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(20, 8))
    plt.grid(True)
    plt.title('Learning curves')
    plt.gca().set_ylim(0, 1)
    plt.show()


# %%
# U-Net
input = Input(shape=(256, 256, 3))

conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(input)
conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
pool1 = Dropout(0.25)(pool1)

conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
pool2 = Dropout(0.5)(pool2)

conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)
pool3 = Dropout(0.5)(pool3)

conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(0.5)(pool4)

convm = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)
convm = Conv2D(1024, (3, 3), activation="relu", padding="same")(convm)

deconv4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = concatenate([deconv4, conv4])
uconv4 = Dropout(0.5)(uconv4)
uconv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(uconv4)


deconv3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = concatenate([deconv3, conv3])
uconv3 = Dropout(0.5)(uconv3)
uconv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(uconv3)


deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = concatenate([deconv2, conv2])
uconv2 = Dropout(0.5)(uconv2)
uconv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv2)

deconv1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = concatenate([deconv1, conv1])
uconv1 = Dropout(0.5)(uconv1)
uconv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv1)

output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

model = Model(input, output)

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.summary()
tf.keras.utils.plot_model(model, to_file='U_Net.png')
# %%
# Train model
BATCH = 32
AT = tf.data.AUTOTUNE
BUFFER = 1000

STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH

train = train.cache().shuffle(BUFFER).batch(BATCH).repeat()
train = train.prefetch(buffer_size=AT)
val = val.batch(BATCH)


early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

history = model.fit(train,
                    validation_data=val,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    epochs=50,
                    callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(history)

# select a validation data batch
img, mask = next(iter(val))
# make prediction
pred = model.predict(img)
plt.figure(figsize=(20, 28))

k = 0
NORM = mpl.colors.Normalize(vmin=0, vmax=58)

for i in pred:
    # plot the predicted mask
    plt.subplot(4, 3, 1 + k * 3)
    i = tf.argmax(i, axis=-1)
    plt.imshow(i, cmap='jet', norm=NORM)
    plt.axis('off')
    plt.title('Prediction')

    # plot the groundtruth mask
    plt.subplot(4, 3, 2 + k * 3)
    plt.imshow(mask[k], cmap='jet', norm=NORM)
    plt.axis('off')
    plt.title('Ground Truth')

    # plot the actual image
    plt.subplot(4, 3, 3 + k * 3)
    plt.imshow(img[k])
    plt.axis('off')
    plt.title('Actual Image')
    k += 1
    if k == 4: break
plt.suptitle('Predition After 50 Epochs (No Fine-tuning)', color='red', size=20)
plt.show()









