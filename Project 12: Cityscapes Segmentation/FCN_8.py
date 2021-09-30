"""
PROJECT 12: Cityscapes Segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: Fully Convolutional Network(FCN)
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training Training Fully Convolutional Network - FCN 8

FCN 8
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
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, \
    BatchNormalization, Permute, TimeDistributed, Bidirectional, GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, \
    SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D, Conv2DTranspose, ReLU, \
    UpSampling2D, Concatenate, Conv2DTranspose, Input, Add, Activation

from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# %%
# PATH & LABEL PROCESS

# Main path
trian_data_path = 'data/cityscapes_data/train/'
val_data_path = 'data/cityscapes_data/val/'

flist = os.listdir(trian_data_path)
img0 = cv2.imread(trian_data_path + flist[0])
plt.imshow(img0)
print(np.shape(img0))
print(len(flist))

# Reading the actual images and forming them into the training dataset
szy, szx, _ = np.shape(img0)
N_ex = 1500
N_bias = 0
x_train = np.zeros((N_ex, szy, int(szx / 2), 3))
y_train = np.zeros((N_ex, szy, int(szx / 2), 3))
k = 0

for f in flist[N_bias:N_bias + N_ex]:
    x_train[k] = cv2.imread(trian_data_path + f)[:, :256] / 256
    y_train[k] = cv2.imread(trian_data_path + f)[:, 256:] / 256
    k = k + 1

# Reading the actual images and forming them into the validation dataset
flist = os.listdir(val_data_path)
img0 = cv2.imread(val_data_path + flist[0])
N_val = 100

szy, szx, _ = np.shape(img0)
x_val = np.zeros((N_val, szy, int(szx / 2), 3))
y_val = np.zeros((N_val, szy, int(szx / 2), 3))
k = 0

for f in flist[0:N_val]:
    x_val[k] = cv2.imread(val_data_path + f)[:, :256] / 256
    y_val[k] = cv2.imread(val_data_path + f)[:, 256:] / 256
    k = k + 1


# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

def learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(20, 8))
    plt.grid(True)
    plt.title('Learning curves')
    plt.gca().set_ylim(0, 1)
    plt.show()


# %%
nClasses = 3
# FCN 8
input = Input(shape=(256, 256, 3))

# CONVOLUTION 1
conv1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(input)
conv1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
# MAX POOLING 1
pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

# CONVOLUTION 2
conv2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(conv2)
# MAX POOLING 2
pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

# CONVOLUTION 3
conv3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
conv3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(conv3)
conv3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(conv3)
# MAX POOLING 3
pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

# CONVOLUTION 4
conv4 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(pool3)
conv4 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(conv4)
conv4 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(conv4)
# MAX POOLING 4
pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

# CONVOLUTION 5
conv5 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(pool4)
conv5 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(conv5)
conv5 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(conv5)
# MAX POOLING 5
pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

# FULLY CONVOLUTIONAL LAYERS

# CONVOLUTION 6
conv6 = Conv2D(filters=4096, kernel_size=7, activation='relu', padding='same')(pool5)
# CONVOLUTION 7
conv7 = Conv2D(filters=4096, kernel_size=1, activation='relu', padding='same')(conv6)

# 4X CONVOLUTION 7
conv7_4 = Conv2DTranspose(nClasses, kernel_size=4, strides=(4, 4), use_bias=False)(conv7)

# 2X # MAX POOLING 4
pool411 = Conv2D(nClasses, (1, 1), activation='relu', padding='same')(pool4)
pool411_2 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(pool411)

pool311 = Conv2D(nClasses, (1, 1), activation='relu', padding='same')(pool3)

output = Add()([pool411_2, pool311, conv7_4])
output = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(output)
output = Activation('softmax')(output)

model = Model(input, output)

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

model.summary()
tf.keras.utils.plot_model(model, to_file='FCN_8.png')
# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

history = model.fit(x_train, y_train,
                    epochs=20,
                    shuffle=True,
                    batch_size=5,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(history)

# %%
# Prediction
prediction_seen = model.predict(x_val[0:20, :, :, :])

ni = 10
for k in range(ni):
    plt.figure(figsize=(10, 30))
    plt.subplot(ni, 3, 1 + k * 3)
    plt.imshow(x_val[k])
    plt.subplot(ni, 3, 2 + k * 3)
    plt.imshow(y_val[k])
    plt.subplot(ni, 3, 3 + k * 3)
    plt.imshow(pp[k])
