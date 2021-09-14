"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL

Development of skills for building and improving simple CNN models for multi-class classification
using the sequential Tensorflow API.

STAGE OBJECTIVES
1. EDA
2. Data preparation
3. Training simple CNN model

DATASET: COVID-19 Radiography Database

ATTRIBUTE INFORMATION:
1. COVID data
2. Normal images
3. Lung opacity images
4. Viral Pneumonia images
"""
# %%
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D, \
    AveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import PrecisionAtRecall, Recall

# %%
# IMPORT DATA

data_path = '/Users/rttrif/Data_Science_Projects/Tensorflow_Certification/Prokect_4_COVID_19_Radiography_Database/' \
            'data/COVID19_Radiography_Dataset'

classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
num_classes = len(classes)


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


def model_confusion_matrix(y_true, y_pred):
    y_true = test_gen.classes

    predictions = np.array(list(map(lambda x: np.argmax(x), y_pred)))

    CMatrix = pd.DataFrame(confusion_matrix(y_true, predictions), columns=classes, index=classes)

    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(CMatrix, annot=True, fmt='g', vmin=0, vmax=250, cmap='Blues')
    ax.set_xlabel('Predicted', fontsize=14, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0);

    ax.set_ylabel('Actual', fontsize=14, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0);
    ax.set_title('Confusion Matrix - Test Set', fontsize=16, weight='bold', pad=20);
    plt.show()


# %%
# DATA PREPROCESSING

BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True, validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  validation_split=0.25)

train_gen = train_datagen.flow_from_directory(directory=data_path,
                                              target_size=(256, 256),
                                              class_mode='categorical',
                                              subset='training',
                                              shuffle=True, classes=classes,
                                              batch_size=BATCH_SIZE,
                                              color_mode="grayscale")

test_gen = test_datagen.flow_from_directory(directory=data_path,
                                            target_size=(256, 256),
                                            class_mode='categorical',
                                            subset='validation',
                                            shuffle=False, classes=classes,
                                            batch_size=BATCH_SIZE,
                                            color_mode="grayscale")


# %%
# Basic model - Tiny VGG

def model_1(epochs):
    model = Sequential([
        Conv2D(10, (3, 3), activation="relu", padding='Same', input_shape=(256, 256, 1)),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(pool_size=2, padding='Same'),

        Conv2D(10, 3, activation='relu'),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(pool_size=2, ),

        Flatten(),
        Dense(num_classes, activation='softmax')

    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1.png')

    history = model.fit(train_gen,
                        steps_per_epoch=len(train_gen) // BATCH_SIZE,
                        validation_steps=len(test_gen) // BATCH_SIZE,
                        validation_data=test_gen,
                        epochs=epochs)

    return history, model


history_1, model_1 = model_1(epochs=10)

# Learning curves
learning_curves(history_1)

# Evaluation model
evaluation_model(history_1)

# Evaluate the model on the test set
evaluate_1 = model_1.evaluate(test_gen, verbose=2)

# Predicting the test set results
y_pred = model_1.predict(test_gen)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
model_confusion_matrix(test_gen, y_pred)


# %%
# Basic model - Tiny VGG
# + Doubling the number of filters

def model_2(epochs):
    model = Sequential([
        Conv2D(20, (3, 3), activation="relu", padding='Same', input_shape=(256, 256, 1)),
        Conv2D(20, 3, activation='relu'),
        MaxPool2D(pool_size=2, padding='Same'),

        Conv2D(20, 3, activation='relu'),
        Conv2D(20, 3, activation='relu'),
        MaxPool2D(pool_size=2, ),

        Flatten(),
        Dense(num_classes, activation='softmax')

    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2.png')

    history = model.fit(train_gen,
                        steps_per_epoch=len(train_gen) // BATCH_SIZE,
                        validation_steps=len(test_gen) // BATCH_SIZE,
                        validation_data=test_gen,
                        epochs=epochs)

    return history, model


history_2, model_2 = model_2(epochs=10)

# Learning curves
learning_curves(history_2)

# Evaluation model
evaluation_model(history_2)

# Evaluate the model on the test set
evaluate_2 = model_1.evaluate(test_gen, verbose=2)

# Predicting the test set results
y_pred = model_2.predict(test_gen)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
model_confusion_matrix(test_gen, y_pred)


# %%
# Basic model - Tiny VGG
# + Doubling the number of filters
# + Adding fully connected layers

def model_3(epochs):
    model = Sequential([
        Conv2D(20, (3, 3), activation="relu", padding='Same', input_shape=(256, 256, 1)),
        Conv2D(20, 3, activation='relu'),
        MaxPool2D(pool_size=2, padding='Same'),

        Conv2D(20, 3, activation='relu'),
        Conv2D(20, 3, activation='relu'),
        MaxPool2D(pool_size=2, ),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')

    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_3.png')

    history = model.fit(train_gen,
                        steps_per_epoch=len(train_gen) // BATCH_SIZE,
                        validation_steps=len(test_gen) // BATCH_SIZE,
                        validation_data=test_gen,
                        epochs=epochs)

    return history, model


history_3, model_3 = model_3(epochs=10)

# Learning curves
learning_curves(history_3)

# Evaluation model
evaluation_model(history_3)

# Evaluate the model on the test set
evaluate_3 = model_1.evaluate(test_gen, verbose=2)

# Predicting the test set results
y_pred = model_3.predict(test_gen)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
model_confusion_matrix(test_gen, y_pred)


# %%
# Basic model - Tiny VGG
# + Doubling the number of filters
# + Adding fully connected layers

def model_4(epochs):
    model = Sequential([
        Conv2D(20, (3, 3), activation="relu", padding='Same', input_shape=(256, 256, 1)),
        Conv2D(20, 3, activation='relu'),
        MaxPool2D(pool_size=2, padding='Same'),

        Conv2D(20, 3, activation='relu'),
        Conv2D(20, 3, activation='relu'),
        MaxPool2D(pool_size=2, ),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')

    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_4.png')

    history = model.fit(train_gen,
                        steps_per_epoch=len(train_gen) // BATCH_SIZE,
                        validation_steps=len(test_gen) // BATCH_SIZE,
                        validation_data=test_gen,
                        epochs=epochs)

    return history, model


history_4, model_4 = model_4(epochs=500)

# Learning curves
learning_curves(history_4)

# Evaluation model
evaluation_model(history_4)

# Evaluate the model on the test set
evaluate_4 = model_1.evaluate(test_gen, verbose=2)

# Predicting the test set results
y_pred = model_4.predict(test_gen)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
model_confusion_matrix(test_gen, y_pred)
