"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models for binary classification
using the sequential and functional Tensorflow API

STAGE OBJECTIVES
2. Training a basic simple model using the sequential Tensorflow API and improving it

DATASET: Heart Failure Prediction
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

import itertools
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import Regularizer

from functools import partial

# %%
# IMPORT DATA

data_path = "/Users/rttrif/Data_Science_Projects/Tensorflow_Certification/" \
            "Project_2_ Heart_Failure_Prediction/data/"

# Read X_train
X_train = pd.read_csv(data_path + '/X_train.csv')
X_train_head = X_train.head()

# Read X_test
X_test = pd.read_csv(data_path + '/X_test.csv')
X_test_head = X_test.head()

# Read y_train
y_train = pd.read_csv(data_path + '/y_train.csv')
y_train_head = y_train.head()

# Read y_test
y_test = pd.read_csv(data_path + '/y_test.csv')
y_test_head = y_test.head()


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
# Basic model

def model_1a(epochs):
    model = Sequential([
        Dense(32, activation='sigmoid', input_shape=(12,), name='model_1a'),
        Dense(16, activation='sigmoid'),
        Dense(8, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1a.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1a = model_1a(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1a = model_1a.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1a.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
def model_1b(epochs):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(12,), name='model_1b'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1b.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1b = model_1b(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1b = model_1b.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1b.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function

def model_1c(epochs):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(12,), name='model_1c'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1c.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1c = model_1c(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1c = model_1c.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1c.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function
# + Increasing the number of hidden layers

def model_1d(epochs):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(12,), name='model_1d'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1d.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1d = model_1d(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1d = model_1d.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1d.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function
# + Increasing the number of hidden layers
# + Increasing the number of neurons in the hidden layers

def model_1e(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(12,), name='model_1e'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1e.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1e = model_1e(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1e = model_1e.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1e.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function
# + Increasing the number of hidden layers
# + Increasing the number of neurons in the hidden layers
# + Applying Dropout

def model_1f(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(12,), name='model_1f'),
        Dropout(rate=0.25),

        Dense(256, activation='relu'),
        Dropout(rate=0.25),

        Dense(128, activation='relu'),
        Dropout(rate=0.25),

        Dense(64, activation='relu'),
        Dropout(rate=0.25),

        Dense(32, activation='relu'),
        Dropout(rate=0.25),

        Dense(16, activation='relu'),
        Dropout(rate=0.25),

        Dense(8, activation='relu'),
        Dropout(rate=0.25),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1f.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1f = model_1f(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1f = model_1f.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1f.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function
# + Increasing the number of hidden layers
# + Increasing the number of neurons in the hidden layers
# + Applying Dropout
# + Applying BatchNormalization

def model_1g(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(12,), name='model_1g'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(256, activation='relu'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(64, activation='relu'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(32, activation='relu'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(16, activation='relu'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(8, activation='relu'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1g.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1g = model_1g(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1g = model_1g.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1g.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function
# + Increasing the number of hidden layers
# + Increasing the number of neurons in the hidden layers
# + Applying Dropout
# + Applying BatchNormalization
# + Applying L2 regularization

def model_1h(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(12,), name='model_1h', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(16, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(8, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1h.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1h = model_1h(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1h = model_1h.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1h.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Basic model
# + Change the activation function in hidden layers
# + Change the optimisation function
# + Increasing the number of hidden layers
# + Increasing the number of neurons in the hidden layers
# + Applying Dropout
# + Applying BatchNormalization
# + Applying L2 regularization
# + Increasing the number of training epochs and early stopping

def model_1i(epochs, patience):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(12,), name='model_1i', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(16, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(8, activation='relu', kernel_regularizer='l2'),
        Dropout(rate=0.25),
        BatchNormalization(),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1i.png')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_1i.h5",
                                                       save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[checkpoint_cb, early_stopping_cb])

    return history, model


history, model_1i = model_1i(epochs=500, patience=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1i = model_1i.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1i.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)

# %%
# COMPARISON OF MODEL RESULTS

# 0 - Basic model
# 1 - Change the activation function in hidden layers
# 2 - Change the optimisation function
# 3 - Increasing the number of hidden layers
# 4 - Increasing the number of neurons in the hidden layers
# 5 - Applying Dropout
# 6 - Applying BatchNormalization
# 7 - Applying L2 regularization
# 8 - Increasing the number of training epochs and early stopping

model_results = [["model_1a", "Basic model", evaluate_1a[0], evaluate_1a[1]],
                 ["model_1b",  "Activation function", evaluate_1b[0], evaluate_1b[1]],
                 ["model_1c",  "Optimisation function",evaluate_1c[0], evaluate_1c[1]],
                 ["model_1d",  "Number of hidden layers",  evaluate_1d[0], evaluate_1d[1]],
                 ["model_1e",  "Number of neurons in the hidden layers",  evaluate_1e[0], evaluate_1e[1]],
                 ["model_1f",  "Dropout",  evaluate_1f[0], evaluate_1f[1]],
                 ["model_1g",  "BatchNormalization", evaluate_1g[0], evaluate_1g[1]],
                 ["model_1h",  "L2 regularization",  evaluate_1h[0], evaluate_1h[1]],
                 ["model_1i",  "Training epochs and early stopping",  evaluate_1i[0], evaluate_1i[1]]]

all_results = pd.DataFrame(model_results, columns=["model", "Improvement",  "Loss", "Accuracy"])

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
all_results[["Loss", "Accuracy"]].plot(ax=ax[0], kind='bar')
all_results[["Loss"]].plot(ax=ax[1], kind='bar')
plt.show()