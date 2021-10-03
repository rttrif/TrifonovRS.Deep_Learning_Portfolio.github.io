"""
PROJECT 14: Leaf disease segmentation
TASK: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: DeepLab
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training DeepLab

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
import glob
import itertools

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, \
    Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D, Input, UpSampling2D

from tensorflow.keras.applications import ResNet50

# %%
# PATH & LABEL PROCESS

# Main path
image_data_path = Path('data/aug_data/images')
mask_data_path = Path('data/aug_data/masks/')

# Listing files
image_path = list(image_data_path.glob(r"*.jpg"))
mask_path = list(mask_data_path.glob(r"*.png"))

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
# ATROUS SPATIAL PYRAMID POOLING
def ASPP(inputs):
    shape = inputs.shape

    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


# %%
# MODEL DeepLab V3+
def DeepLabV3Plus(shape):
    # Inputs
    inputs = Input(shape)

    # Pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Pre-trained ResNet50 Output
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    # Get low-level features
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    # Outputs
    x = Conv2D(1, (1, 1), name='output_layer')(x)
    x = Activation('sigmoid')(x)

    # Model
    model = Model(inputs=inputs, outputs=x)
    return model


input_shape = (256, 256, 3)
model = DeepLabV3Plus(input_shape)
model.summary()
tf.keras.utils.plot_model(model, to_file='DeepLabV3_Plus.png')
# %%
# Train model
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

history = model.fit(transformation_image, transformation_mask,
                    epochs=20,
                    shuffle=True,
                    batch_size=5,
                    validation_split=0.1,
                    callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(history)