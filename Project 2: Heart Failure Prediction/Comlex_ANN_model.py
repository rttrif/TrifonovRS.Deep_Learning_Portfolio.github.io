"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models for binary classification
using the sequential and functional Tensorflow API

STAGE OBJECTIVES
3. Training the basic complex model using the Tensorflow functional API and improving it.

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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
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
# Complex Basic model: 4 branches - 1 wide branch and 3 deep branches
# Deep branch 1: elu activation function
# Deep branch 2: selu activation function
# Deep branch 3: relu activation function


def model_2a(epochs):
    input = Input(shape=(12,), name='model_2a')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu')(input)
    branch_12 = Dense(16, activation='elu')(branch_11)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu')(input)
    branch_22 = Dense(72, activation='selu')(branch_21)
    branch_23 = Dense(36, activation='selu')(branch_22)
    branch_24 = Dense(12, activation='selu')(branch_23)

    # Deep branch 3
    branch_31 = Dense(64, activation='relu')(input)
    branch_32 = Dense(64, activation='relu')(branch_31)
    branch_33 = Dense(128, activation='relu')(branch_32)
    branch_34 = Dense(128, activation='relu')(branch_33)
    branch_35 = Dense(64, activation='relu')(branch_34)
    branch_36 = Dense(64, activation='relu')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='model_2a')

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2a.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model_2a = model_2a(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2a = model_2a.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2a.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Complex Basic model: 4 branches - 1 wide branch and 3 deep branches
# Deep branch 1: elu activation function
# Deep branch 2: selu activation function
# Deep branch 3: relu activation function
# + Change the optimisation function to Nadam


def model_2b(epochs):
    input = Input(shape=(12,), name='model_2b')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu')(input)
    branch_12 = Dense(16, activation='elu')(branch_11)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu')(input)
    branch_22 = Dense(72, activation='selu')(branch_21)
    branch_23 = Dense(36, activation='selu')(branch_22)
    branch_24 = Dense(12, activation='selu')(branch_23)

    # Deep branch 3
    branch_31 = Dense(64, activation='relu')(input)
    branch_32 = Dense(64, activation='relu')(branch_31)
    branch_33 = Dense(128, activation='relu')(branch_32)
    branch_34 = Dense(128, activation='relu')(branch_33)
    branch_35 = Dense(64, activation='relu')(branch_34)
    branch_36 = Dense(64, activation='relu')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='model_2b')

    model.compile(optimizer='Nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2b.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model_2b = model_2b(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2b = model_2b.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2b.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# Complex Basic model: 4 branches - 1 wide branch and 3 deep branches
# Deep branch 1: elu activation function
# Deep branch 2: selu activation function
# Deep branch 3: relu activation function
# + Change the optimisation function to Nadam
# + Change the optimisation function by Nadam to RMSprop


def model_2c(epochs):
    input = Input(shape=(12,), name='model_2c')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu')(input)
    branch_12 = Dense(16, activation='elu')(branch_11)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu')(input)
    branch_22 = Dense(72, activation='selu')(branch_21)
    branch_23 = Dense(36, activation='selu')(branch_22)
    branch_24 = Dense(12, activation='selu')(branch_23)

    # Deep branch 3
    branch_31 = Dense(64, activation='relu')(input)
    branch_32 = Dense(64, activation='relu')(branch_31)
    branch_33 = Dense(128, activation='relu')(branch_32)
    branch_34 = Dense(128, activation='relu')(branch_33)
    branch_35 = Dense(64, activation='relu')(branch_34)
    branch_36 = Dense(64, activation='relu')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='model_2c')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2c.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model_2c = model_2c(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2c = model_2c.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2c.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
# COMPLEX BASIC MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
# Deep branch 1: elu activation function
# Deep branch 2: selu activation function
# Deep branch 3: relu activation function
# + Change the optimisation function to Nadam
# + Change the optimisation function by Nadam to RMSprop
# + APPLYING REGULARIZATION
# Deep branch 1: L1
# Deep branch 2: L2
# Deep branch 3: L1 and L2


def model_2d(epochs):
    input = Input(shape=(12,), name='model_2d')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(branch_21)
    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(branch_22)
    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(branch_23)

    # Deep branch 3
    branch_31 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(input)
    branch_32 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(branch_31)
    branch_33 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(branch_32)
    branch_34 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(branch_33)
    branch_35 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(branch_34)
    branch_36 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='model_2d')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2d.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model_2d = model_2d(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2d = model_2d.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2d.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)
# %%
# COMPLEX BASIC MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
# Deep branch 1: elu activation function
# Deep branch 2: selu activation function
# Deep branch 3: relu activation function
# + Change the optimisation function to Nadam
# + Change the optimisation function by Nadam to RMSprop
# + APPLYING REGULARIZATION
# Deep branch 1: L1
# Deep branch 2: L2
# Deep branch 3: L1 and L2
# + Applying BatchNormalization


def model_2e(epochs):
    input = Input(shape=(12,), name='model_2e')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)
    out_branch_1 = BatchNormalization()(branch_12)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    bach_norm = BatchNormalization()(branch_21)

    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_22)

    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_23)

    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(bach_norm)
    out_branch_2 = BatchNormalization()(branch_24)

    # Deep branch 3
    branch_31 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(input)
    bach_norm = BatchNormalization()(branch_31)

    branch_32 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_32)

    branch_33 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_33)

    branch_34 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_34)

    branch_35 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_35)

    branch_36 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    out_branch_3 = BatchNormalization()(branch_36)

    concat = Concatenate()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='model_2e')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2e.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model_2e = model_2e(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2e = model_2e.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2e.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)

# %%
# COMPLEX BASIC MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
# Deep branch 1: elu activation function
# Deep branch 2: selu activation function
# Deep branch 3: relu activation function
# + Change the optimisation function to Nadam
# + Change the optimisation function by Nadam to RMSprop
# + APPLYING REGULARIZATION
# Deep branch 1: L1
# Deep branch 2: L2
# Deep branch 3: L1 and L2
# + Applying BatchNormalization
# + Applying Dropout


def model_2f(epochs):
    input = Input(shape=(12,), name='model_2f')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)
    dropout = Dropout(rate=0.05)(branch_12)
    out_branch_1 = BatchNormalization()(dropout)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    bach_norm = BatchNormalization()(branch_21)

    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_22)

    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_23)

    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.25)(branch_24)
    out_branch_2 = BatchNormalization()(dropout)

    # Deep branch 3
    branch_31 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(input)
    bach_norm = BatchNormalization()(branch_31)

    branch_32 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_32)

    branch_33 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_33)

    branch_34 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_34)

    branch_35 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_35)

    branch_36 = Dense(64, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    dropout = Dropout(rate=0.50)(branch_36)
    out_branch_3 = BatchNormalization()(dropout)

    concat = Concatenate()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='model_2f')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2f.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model_2f = model_2f(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2f = model_2f.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2f.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)

# %%
# COMPARISON OF MODEL RESULTS
"""
0 - COMPLEX BASIC MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
    - Deep branch 1: elu activation function
    - Deep branch 2: selu activation function
    - Deep branch 3: relu activation function
    
1- Change the optimisation function to Nadam
2- Change the optimisation function by Nadam to RMSprop

3- APPLYING REGULARIZATION
    - Deep branch 1: L1
    - Deep branch 2: L2
    - Deep branch 3: L1 and L2
4- Applying BatchNormalization
5- Applying Dropout
"""

model_results = [["model_2a", "Basic model", evaluate_2a[0], evaluate_2a[1]],
                 ["model_2b",  "Nadam", evaluate_2b[0], evaluate_2b[1]],
                 ["model_2c",  "RMSprop",evaluate_2c[0], evaluate_2c[1]],
                 ["model_2d",  "Regularization",  evaluate_2d[0], evaluate_2d[1]],
                 ["model_2e",  "BatchNormalization",  evaluate_2e[0], evaluate_2e[1]],
                 ["model_2f",  "Dropout",  evaluate_2f[0], evaluate_2f[1]]]


all_results = pd.DataFrame(model_results, columns=["model", "Improvement",  "Loss", "Accuracy"])

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
all_results[["Loss", "Accuracy"]].plot(ax=ax[0], kind='bar')
all_results[["Loss"]].plot(ax=ax[1], kind='bar')
plt.show()
