"""
PROJECT 6: Vehicle Classification

TASK: Classification

PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
> Studying architecture: AlexNet

PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training AlexNet
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
data_path = Path('data')

# Path process
img_path = list(data_path.glob(r"*/*.png"))

# Mapping the labels
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path))

# Transformation to series structure
img_path_series = pd.Series(img_path, name="PICTURE", dtype='object').astype(str)
img_labels_series = pd.Series(img_labels, name="CATEGORY", dtype='object')

# Concatenating series to train_data dataframe
train_df = pd.concat([img_path_series, img_labels_series], axis=1)

# Shuffling
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
# DATA PREPARATION

# Splitting train and test
train_data, test_data = train_test_split(train_df, train_size=0.85, random_state=42)

print("Train shape: ", train_data.shape)
print("Test shape: ", test_data.shape)

print(train_data["CATEGORY"].value_counts())
print(test_data["CATEGORY"].value_counts())

# Converting the label to a numeric format
y_test = LabelEncoder().fit_transform(test_data["CATEGORY"])

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
                                     validation_split=0.1)

test_generator = ImageDataGenerator(rescale=1. / 255)

# %%
# Applying generator and transformation to tensor
print("Preparing the training dataset:")
train_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="PICTURE",
                                                   y_col="CATEGORY",
                                                   target_size=(75, 75),
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   batch_size=32,
                                                   subset="training")

print("Preparing the validation dataset:")
valid_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="PICTURE",
                                                   y_col="CATEGORY",
                                                   target_size=(75, 75),
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   batch_size=32,
                                                   subset="validation")
print("Preparing the test dataset:")
test_images = test_generator.flow_from_dataframe(dataframe=test_data,
                                                 x_col="PICTURE",
                                                 y_col="CATEGORY",
                                                 target_size=(75, 75),
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=32)

# Checking
print("Checking the training dataset:")
for data_batch, label_batch in train_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

print("Checking the validation dataset:")
for data_batch, label_batch in valid_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

print("Checking the test dataset:")
for data_batch, label_batch in test_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break


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
# MODEL: AlexNet

def AlexNet(epochs, patience):
    model = Sequential([
        # Layer C1: Convolution Layer (96, 11×11)
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(75, 75, 3)),
        # Layer S2: Max Pooling Layer (3×3)
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer C3: Convolution Layer (256, 5×5)
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        # Layer S4: Max Pooling Layer (3×3)
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer C5: Convolution Layer (384, 3×3)
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        # Layer C6: Convolution Layer (384, 3×3)
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        # Layer C7: Convolution Layer (256, 3×3)
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        # Layer S8: Max Pooling Layer (3×3)
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # Flatten
        Flatten(),
        # Layer F9: Fully-Connected Layer (4096)
        Dense(4096, activation='relu'),
        # Layer F10: Fully-Connected Layer (4096)
        Dense(4096, activation='relu'),
        # Layer F11: Fully-Connected Layer (1000)
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='AlexNet_base_model.png')

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)
    history = model.fit(train_images,
                        validation_data=valid_images,
                        epochs=epochs,
                        callbacks=[early_stopping_cb])

    return history, model


history, model = AlexNet(epochs=5, patience=5)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate = model.evaluate(test_images, verbose=2)

# Predicting the test set results
y_pred = model.predict(test_images)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
model_confusion_matrix(y_test, y_pred_class)
# %%
# MODEL: AlexNet
# Improving the basic model by applying regularization strategies:
# Add BatchNormalization and Dropout
def AlexNet_IMP(epochs, patience):
    model = Sequential([
        # Layer C1: Convolution Layer (96, 11×11)
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(75, 75, 3)),
        BatchNormalization(),
        # Layer S2: Max Pooling Layer (3×3)
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer C3: Convolution Layer (256, 5×5)
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        # Layer S4: Max Pooling Layer (3×3)
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # Layer C5: Convolution Layer (384, 3×3)
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        # Layer C6: Convolution Layer (384, 3×3)
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        # Layer C7: Convolution Layer (256, 3×3)
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        # Layer S8: Max Pooling Layer (3×3)
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        # Flatten
        Flatten(),
        # Layer F9: Fully-Connected Layer (4096)
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # Layer F10: Fully-Connected Layer (4096)
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # Layer F11: Fully-Connected Layer (1000)
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='AlexNet_base_model.png')

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)
    history = model.fit(train_images,
                        validation_data=valid_images,
                        epochs=epochs,
                        callbacks=[early_stopping_cb])

    return history, model


history, model = AlexNet_IMP(epochs=500, patience=5)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate = model.evaluate(test_images, verbose=2)

# Predicting the test set results
y_pred = model.predict(test_images)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
model_confusion_matrix(y_test, y_pred_class)