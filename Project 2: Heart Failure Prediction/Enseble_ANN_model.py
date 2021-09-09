"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models for binary classification
using the sequential and functional Tensorflow API

STAGE OBJECTIVES
4. Training an ensemble of several models

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
from tensorflow.keras.layers import Dense, Input, Concatenate, Average, Maximum, Minimum, Multiply
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
# CONCATENATE MODELS

# COMPLEX MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
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


def concatenate_models(epochs):
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


history, concatenate_models = concatenate_models(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_concatenate_models = concatenate_models.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = concatenate_models.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
#  AVERAGE MODELS

# COMPLEX MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
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


def average_ensemble(epochs):
    input = Input(shape=(12,), name='average_ensemble')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)
    dropout = Dropout(rate=0.05)(branch_12)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_1 = Dense(1, activation='sigmoid')(bathc_norm)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    bach_norm = BatchNormalization()(branch_21)

    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_22)

    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_23)

    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.25)(branch_24)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_2 = Dense(1, activation='sigmoid')(bathc_norm)

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
    bathc_norm = BatchNormalization()(dropout)
    out_branch_3 = Dense(1, activation='sigmoid')(bathc_norm)

    concat = Average()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='average_ensemble')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='average_ensemble.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, average_ensemble = average_ensemble(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_average_ensemble = average_ensemble.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = average_ensemble.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
#  MAXIMUM MODELS

# COMPLEX MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
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

def maximum_ensemble(epochs):
    input = Input(shape=(12,), name='maximum_ensemble')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)
    dropout = Dropout(rate=0.05)(branch_12)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_1 = Dense(1, activation='sigmoid')(bathc_norm)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    bach_norm = BatchNormalization()(branch_21)

    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_22)

    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_23)

    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.25)(branch_24)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_2 = Dense(1, activation='sigmoid')(bathc_norm)

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
    bathc_norm = BatchNormalization()(dropout)
    out_branch_3 = Dense(1, activation='sigmoid')(bathc_norm)

    concat = Maximum()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='maximum_ensemble')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='maximum_ensemble.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, maximum_ensemble = maximum_ensemble(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_maximum_ensemble = maximum_ensemble.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = maximum_ensemble.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
#  MINIMUM MODELS

# COMPLEX MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
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

def minimum_ensemble(epochs):
    input = Input(shape=(12,), name='minimum_ensemble')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)
    dropout = Dropout(rate=0.05)(branch_12)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_1 = Dense(1, activation='sigmoid')(bathc_norm)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    bach_norm = BatchNormalization()(branch_21)

    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_22)

    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_23)

    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.25)(branch_24)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_2 = Dense(1, activation='sigmoid')(bathc_norm)

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
    bathc_norm = BatchNormalization()(dropout)
    out_branch_3 = Dense(1, activation='sigmoid')(bathc_norm)

    concat = Minimum()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='minimum_ensemble')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='minimum_ensemble.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, minimum_ensemble = minimum_ensemble(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_minimum_ensemble = minimum_ensemble.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = minimum_ensemble.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)


# %%
#  MULTIPLY MODELS

# COMPLEX MODEL: 4 BRANCHES - 1 WIDE BRANCH AND 3 DEEP BRANCHES:
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

def multiply_ensemble(epochs):
    input = Input(shape=(12,), name='multiply_ensemble')

    # Deep branch 1
    branch_11 = Dense(32, activation='elu', kernel_regularizer='l1')(input)
    branch_12 = Dense(16, activation='elu', kernel_regularizer='l1')(branch_11)
    dropout = Dropout(rate=0.05)(branch_12)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_1 = Dense(1, activation='sigmoid')(bathc_norm)

    # Deep branch 2
    branch_21 = Dense(144, activation='selu', kernel_regularizer='l2')(input)
    bach_norm = BatchNormalization()(branch_21)

    branch_22 = Dense(72, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_22)

    branch_23 = Dense(36, activation='selu', kernel_regularizer='l2')(bach_norm)
    bach_norm = BatchNormalization()(branch_23)

    branch_24 = Dense(12, activation='selu', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.25)(branch_24)
    bathc_norm = BatchNormalization()(dropout)
    out_branch_2 = Dense(1, activation='sigmoid')(bathc_norm)

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
    bathc_norm = BatchNormalization()(dropout)
    out_branch_3 = Dense(1, activation='sigmoid')(bathc_norm)

    concat = Multiply()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='sigmoid')(concat)

    model = Model(inputs=input, outputs=output, name='multiply_ensemble')

    model.compile(optimizer='RMSprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='multiply_ensemble.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=16,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, multiply_ensemble = multiply_ensemble(epochs=20)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_multiply_ensemble = multiply_ensemble.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = multiply_ensemble.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()

# Confusion matrix
model_confusion_matrix(y_test, y_pred)

# %%
# COMPARISON OF MODEL RESULTS
Concatenate, Average, Maximum, Minimum, Multiply

model_results = [["concatenate_ensemble", "Concatenate", evaluate_concatenate_models[0], evaluate_concatenate_models[1]],
                 ["average_ensemble", "Average", evaluate_average_ensemble[0], evaluate_average_ensemble[1]],
                 ["maximum_ensemble", "Maximum", evaluate_maximum_ensemble[0], evaluate_maximum_ensemble[1]],
                 ["minimum_ensemble", "Minimum", evaluate_minimum_ensemble[0], evaluate_minimum_ensemble[1]],
                 ["multiply_ensemble", "Multiply", evaluate_multiply_ensemble[0], evaluate_multiply_ensemble[1]]]

all_results = pd.DataFrame(model_results, columns=["model", "Ensemble", "Loss", "Accuracy"])

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
all_results[["Loss", "Accuracy"]].plot(ax=ax[0], kind='bar')
all_results[["Loss"]].plot(ax=ax[1], kind='bar')
plt.show()
