"""
PROJECT 5: Forest Fire Detection

TASK: Classification

PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
> Studying architecture: LeNet-5

PROJECT OBJECTIVES

1. Exploratory Data Analysis
2. Training LeNet-5
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
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, \
    AveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.preprocessing import image

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import PrecisionAtRecall, Recall

# %%
# PATH & LABEL PROCESS

# Main path
data_path = Path('data/fire_dataset')

# Path process
img_path = list(data_path.glob(r"*/*.png"))

# Label process
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path))

print("FIRE: ", img_labels.count("fire_images"))
print("NO_FIRE: ", img_labels.count("non_fire_images"))

# %%
# Evaluation and visualization of model parameters

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
# TRANSFORMATION TO SERIES STRUCTURE
img_path_series = pd.Series(img_path, name="PNG").astype(str)
img_labels_series = pd.Series(img_labels, name="CATEGORY")

print(img_path_series)
print(img_labels_series)

img_labels_series.replace({"non_fire_images": "NO_FIRE", "fire_images": "FIRE"}, inplace=True)

print(img_labels_series)

# %%
# TRANSFORMATION TO DATAFRAME STRUCTURE

train_df = pd.concat([img_path_series, img_labels_series], axis=1)

print(train_df.head(-1))

print(train_df.info())

# %%
# Shuffling
train_df = train_df.sample(frac=1).reset_index(drop=True)
print(train_df.head(-1))

# %%
# DATA PREPARATION

# Image generator
train_generator = ImageDataGenerator(rescale=1. / 255,
                                     shear_range=0.3,
                                     zoom_range=0.2,
                                     brightness_range=[0.2, 0.9],
                                     rotation_range=30,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode="nearest",
                                     validation_split=0.15)

test_generator = ImageDataGenerator(rescale=1. / 255)

# %%
# Splitting train and test
train_data, test_data = train_test_split(train_df, train_size=0.85, random_state=42, shuffle=True)

print("Train shape: ", train_data.shape)
print("Test shape: ", test_data.shape)
print(test_data["CATEGORY"].value_counts())

# %%
# Applying generator and transformation to tensor
train_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="PNG",
                                                   y_col="CATEGORY",
                                                   target_size=(256, 256),
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="training")

valid_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="PNG",
                                                   y_col="CATEGORY",
                                                   target_size=(256, 256),
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="validation")

test_images = test_generator.flow_from_dataframe(dataframe=test_data,
                                                 x_col="PNG",
                                                 y_col="CATEGORY",
                                                 target_size=(256, 256),
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=32)

# %%
# Checking

for data_batch, label_batch in train_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

for data_batch, label_batch in valid_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

for data_batch, label_batch in test_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break


# %%
# MODEL: LeNet-5

def LeNet_5(epochs):
    model = Sequential([
        # C1
        Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=(256, 256, 3), padding='same'),
        # S2
        AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
        # C3
        Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'),
        # S4
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Flatten
        Flatten(),
        # C5
        Dense(120, activation='tanh'),
        # F6
        Dense(84, activation='tanh'),
        # Output layer
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='LeNet_5.png')

    history = model.fit(train_images,
                        validation_data=valid_images,
                        epochs=epochs)

    return history, model


history, model = LeNet_5(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1 = model.evaluate(test_images, verbose=2)

# Predicting the test set results
y_pred = model.predict(test_images)
y_pred_class = np.argmax(y_pred, axis=1)

prediction_class = LabelEncoder().fit_transform(test_data["CATEGORY"])

# Confusion matrix
model_confusion_matrix(prediction_class, y_pred_class)

