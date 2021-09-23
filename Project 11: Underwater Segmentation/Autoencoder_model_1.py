"""
PROJECT 11: Underwater Segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: Autoencoder for semantic segmentation
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training Autoencoder

MODEL 1
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
# PATH & LABEL PROCESS

# Main path
image_data_path = Path('data/train_val/images')
mask_data_path = Path('data/train_val/masks')

# Listing files
image_path = list(image_data_path.glob(r"*.jpg"))
mask_path = list(mask_data_path.glob(r"*.bmp"))

# Sort files
image_path = sorted(image_path)
mask_path = sorted(mask_path)

# Transformation to series
image_series = pd.Series(image_path, name="image", dtype='object').astype(str)
mask_series = pd.Series(mask_path, name="mask", dtype='object').astype(str)

# Concatenating series to train_data dataframe
train_df = pd.concat([image_series, mask_series], axis=1)

# Checking results
print(train_df.info())
# %%
# DATA PREPARATION
image_main_transformation = []
mask_main_transformation = []
add_main_transformation = []

for image, mask in zip(train_df['image'], train_df['mask']):
    image_x = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    mask_x = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)

    resized_x_image = cv2.resize(image_x, (256, 256))
    resized_x_mask = cv2.resize(mask_x, (256, 256))

    add_x = cv2.addWeighted(resized_x_image, 0.6, resized_x_mask, 0.6, 0.5)

    resized_x_add = cv2.resize(add_x, (256, 256))

    image_main_transformation.append(resized_x_image)
    mask_main_transformation.append(resized_x_mask)
    add_main_transformation.append(resized_x_add)
# %%
# Checking
print(f'Array image shape: {np.shape(np.array(image_main_transformation))}')
print(f'Array mask shape: {np.shape(np.array(mask_main_transformation))}')
print(f'Array add shape: {np.shape(np.array(add_main_transformation))}')

ind = np.random.randint(0, train_df["image"].shape[0])

figure, axis = plt.subplots(1, 3, figsize=(15, 15))

axis[0].imshow(image_main_transformation[ind], cmap="jet")
axis[0].set_xlabel(image_main_transformation[ind].shape)
axis[0].set_title("Original")

axis[1].imshow(mask_main_transformation[ind])
axis[1].set_xlabel(mask_main_transformation[ind].shape)
axis[1].set_title("Mask")

axis[2].imshow(add_main_transformation[ind])
axis[2].set_xlabel(add_main_transformation[ind].shape)
axis[2].set_title("Add")

plt.tight_layout()
plt.show()

# %%
# Transformation
transformation_image = np.array(image_main_transformation, dtype="float32")
transformation_mask = np.array(mask_main_transformation, dtype="float32")
transformation_add = np.array(add_main_transformation, dtype="float32")

transformation_image = transformation_image / 255.
transformation_mask = transformation_mask / 255.
transformation_add = transformation_add / 255.

print("Train: ", transformation_image.shape)
print("Transformation mask: ", transformation_mask.shape)
print("Transformation add: ", transformation_add.shape)


# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

def learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(20, 8))
    plt.grid(True)
    plt.title('Learning curves')
    plt.gca().set_ylim(0, 1)
    plt.show()

# %%
# AUTOENCODER MODEL 1

input = Input(shape=(256, 256, 3))

# CONVOLUTION
# Conv 1
conv = Conv2D(filters=32, kernel_size=5, activation='relu')(input)
conv = BatchNormalization()(conv)

# Conv 2
conv = Conv2D(filters=64, kernel_size=5, activation='relu')(conv)
conv = BatchNormalization()(conv)

# Conv 3
conv = Conv2D(filters=128, kernel_size=5, activation='relu')(conv)
conv = BatchNormalization()(conv)

# Conv 4
conv = Conv2D(filters=256, kernel_size=5, activation='relu')(conv)
conv = BatchNormalization()(conv)

# DECONVOLUTION
# Deconv 1
deconv = Conv2DTranspose(filters=128, kernel_size=5, activation='relu')(conv)

# Deconv 2
deconv = Conv2DTranspose(filters=64, kernel_size=5, activation='relu')(deconv)

# Deconv 3
deconv = Conv2DTranspose(filters=32, kernel_size=5, activation='relu')(deconv)

# Output
output = Conv2DTranspose(filters=3, kernel_size=5, activation='sigmoid')(deconv)

model = Model(inputs=input, outputs=output)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=["mse"])
model.summary()
tf.keras.utils.plot_model(model, to_file='Autoencoder_model_1.png')

# %%
# Train model


early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

history = model.fit(transformation_image, transformation_mask,
                    batch_size=2,
                    epochs=10,
                    callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(history)

# %%
# Prediction
prediction_seen = model.predict(transformation_image[:10])

figure, axis = plt.subplots(1, 2, figsize=(15, 15))

pre_count = 7

axis[0].imshow(transformation_image[pre_count])
axis[0].set_title("Original")
axis[1].imshow(prediction_seen[pre_count])
axis[1].set_title("Prediction")

plt.tight_layout()
plt.show()
