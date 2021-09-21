"""
PROJECT 8: Weather Classification
TASK: Multi-class classification
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: ResNet
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training ResNet-34
3. Training ResNet-50
4. Training ResNet-101
"""
# %%
# IMPORT LIBRARIES

# GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, MaxPool2D, GlobalAvgPool2D, Dense
from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# %%
# PATH & LABEL PROCESS

# Main path
data_path = Path('data/Weather_Dataset')

# Listing subdirectories
file_path = list(data_path.glob('**/*.jpg'))

# Mapping the labels
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_path))

# Transformation to series
files = pd.Series(file_path, name="files", dtype='object').astype(str)
labels = pd.Series(img_labels, name="category", dtype='object')

# Concatenating series to train_data dataframe
train_df = pd.concat([files, labels], axis=1)

# Shuffling
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# %%
# DATA PREPARATION

# Splitting train and test
train_data, test_data = train_test_split(train_df, train_size=0.85, random_state=42)

print("Train shape: ", train_data.shape)
print("Test shape: ", test_data.shape)

print(train_data["category"].value_counts())
print(test_data["category"].value_counts())

# Converting the label to a numeric format
test_images = LabelEncoder().fit_transform(test_data["category"])

# %%
# Image generator
train_generator = ImageDataGenerator(rescale=1. / 255,
                                     shear_range=0.3,
                                     zoom_range=0.2,
                                     rotation_range=30,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15)

test_generator = ImageDataGenerator(rescale=1. / 255)
# %%
# Applying generator and transformation to tensor
print("Preparing the training data:")
train_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="files",
                                                   y_col="category",
                                                   target_size=(256, 256),
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="training")

print("Preparing the validation data:")
valid_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="files",
                                                   y_col="category",
                                                   target_size=(256, 256),
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="validation")
print("Preparing the test data:")
test_images = test_generator.flow_from_dataframe(dataframe=test_data,
                                                 x_col="files",
                                                 y_col="category",
                                                 target_size=(256, 256),
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=32)
# %%
# Checking
print("Checking the training data:")
for data_batch, label_batch in train_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

print("Checking the validation data:")
for data_batch, label_batch in valid_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

print("Checking the test data:")
for data_batch, label_batch in test_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break


# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

def learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(20, 8))
    plt.grid(True)
    plt.title('Learning curves')
    plt.gca().set_ylim(0, 1)
    plt.show()


def evaluation_model(history):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(20, 8))
    axL.plot(history.history['loss'], label="Training loss")
    axL.plot(history.history['val_loss'], label="Validation loss")
    axL.set_title('Training and Validation loss')
    axL.set_xlabel('Epochs')
    axL.set_ylabel('Loss')
    axL.legend(loc='upper right')

    axR.plot(history.history['accuracy'], label="Training accuracy")
    axR.plot(history.history['val_accuracy'], label="Validation accuracy")
    axR.set_title('Training and Validation accuracy')
    axR.set_xlabel('Epoch')
    axR.set_ylabel('Accuracy')
    axR.legend(loc='upper right')

    plt.show()


def model_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
    plt.show()


# %%
# MODEL: ResNet-50
"""
The network consists of several repeating blocks. 
We will first create auxiliary functions of key blocks to simplify the design of the entire network model.
1. Conv-BatchNorm-ReLU block
2. Identity block
3. Projection block
4. Resnet block
"""


# Conv-BatchNorm-ReLU block
def conv_batchnorm_relu(x, filters, kernel_size, strides):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


# Identity block
def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)

    x = Add()([x, tensor])
    x = ReLU()(x)
    return x


# Projection block
def projection_block(tensor, filters, strides):
    # left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)

    # right stream
    shortcut = Conv2D(filters=4 * filters, kernel_size=1, strides=strides)(tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


# ResNet block
def resnet_block(x, filters, reps, strides):
    x = projection_block(x, filters=filters, strides=strides)
    for _ in range(reps - 1):
        x = identity_block(x, filters=filters)
    return x


# %%
# Model ResNet-50
input = Input(shape=(256, 256, 3))

# conv1: 7x7, 64, strides 2
x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)

# conv2
# 3x3 max pool, strides 2
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
x = resnet_block(x, filters=64, reps=3, strides=2)

# conv3
x = resnet_block(x, filters=128, reps=4, strides=2)

# conv4
x = resnet_block(x, filters=256, reps=6, strides=2)

# conv5
x = resnet_block(x, filters=512, reps=3, strides=2)

# average pool
x = GlobalAvgPool2D()(x)

# fully - connected layer(1000)
x = Dense(1000, activation='relu')(x)

output = Dense(4, activation='softmax')(x)

model = Model(input, output)

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

model.summary()
tf.keras.utils.plot_model(model, to_file='ResNet-50.png')
# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

history = model.fit(train_images,
                    validation_data=valid_images,
                    epochs=5,
                    callbacks=[early_stopping_cb])

# %%
# EVALUATION RESULT
# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1 = model.evaluate(test_images, verbose=2)

# Predicting the test set results
y_pred = model.predict(test_images)
y_pred_class = np.argmax(y_pred, axis=1)

prediction_class = LabelEncoder().fit_transform(test_data["category"])

# Confusion matrix
model_confusion_matrix(prediction_class, y_pred_class)