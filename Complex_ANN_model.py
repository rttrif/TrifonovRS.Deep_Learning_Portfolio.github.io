"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models
using the sequential and functional Tensorflow API

PROJECT OBJECTIVES
Training the basic complex model using the Tensorflow functional API and improving it using various methods.
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from functools import partial

# %%
# IMPORT DATA

data_path = "/Users/rttrif/Data_Science_Projects/Tensorflow_Certification/" \
            "Prokect_1_Household_Electric_Power_Consumption/data"

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
# NORMALIZE DATA

# scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalize X_train
X_train_norm = scaler.fit_transform(X_train)
print(X_train.shape)

# Normalize X_test
X_test_norm = scaler.fit_transform(X_test)
print(X_test.shape)


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
    axL.plot(history.history['loss'], label="loss (mse) for training")
    axL.plot(history.history['val_loss'], label="loss (mse) for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

    axR.plot(history.history['mae'], label="mae for training")
    axR.plot(history.history['val_mae'], label="mae for validation")
    axR.set_title('model mae')
    axR.set_xlabel('epoch')
    axR.set_ylabel('mae')
    axR.legend(loc='upper right')

    plt.show()


# %%
# Basic model with three fully connected layers



# %%
# 0 - Basic model with three fully connected layers
# 1 - Increasing validation_split to 0.4
# 2 - Adding two additional layers
# 3 - Increasing the number of neurons in the hidden layers
# 4 - Changing the activation function
# 5 - Changing the optimization function
# 6 - Increasing the number of training epochs
# 7 - Applying an early stopping and saving the best model
# 8 - Applying L1 and L2 regularization
# 9 - Applying Dropout
# 10 - Applying BatchNormalization


# Evaluate the model on the test set
evaluate_1a = model_1a.evaluate(X_test, y_test, verbose=2)
evaluate_1b = model_1b.evaluate(X_test, y_test, verbose=2)
evaluate_1c = model_1c.evaluate(X_test, y_test, verbose=2)
evaluate_1d = model_1d.evaluate(X_test, y_test, verbose=2)
evaluate_1e = model_1e.evaluate(X_test, y_test, verbose=2)
evaluate_1f = model_1f.evaluate(X_test, y_test, verbose=2)
evaluate_1g = model_1g.evaluate(X_test, y_test, verbose=2)
evaluate_1h = model_1h.evaluate(X_test, y_test, verbose=2)
evaluate_1i = model_1i.evaluate(X_test, y_test, verbose=2)
evaluate_1j = model_1j.evaluate(X_test, y_test, verbose=2)
evaluate_1k = model_1k.evaluate(X_test, y_test, verbose=2)

model_results = [["model_1a", "Basic model with three fully connected layers", model_rmse(model_1a)[0], model_rmse(model_1a)[1], evaluate_1a[0], evaluate_1a[1]],
                 ["model_1b",  "Increasing validation_split to 0.4", model_rmse(model_1b)[0], model_rmse(model_1b)[1], evaluate_1b[0], evaluate_1b[1]],
                 ["model_1c",  "Adding two additional layers", model_rmse(model_1c)[0], model_rmse(model_1c)[1], evaluate_1c[0], evaluate_1c[1]],
                 ["model_1d",  "Increasing the number of neurons in the hidden layers", model_rmse(model_1d)[0], model_rmse(model_1d)[1], evaluate_1d[0], evaluate_1d[1]],
                 ["model_1e",  "Changing the activation function", model_rmse(model_1e)[0], model_rmse(model_1e)[1], evaluate_1e[0], evaluate_1e[1]],
                 ["model_1f",  "Changing the optimization function", model_rmse(model_1f)[0], model_rmse(model_1f)[1], evaluate_1f[0], evaluate_1f[1]],
                 ["model_1g",  "Increasing the number of training epochs", model_rmse(model_1g)[0], model_rmse(model_1g)[1], evaluate_1g[0], evaluate_1g[1]],
                 ["model_1h",  "Applying an early stopping and saving the best model", model_rmse(model_1h)[0], model_rmse(model_1h)[1], evaluate_1h[0], evaluate_1h[1]],
                 ["model_1i",  "Applying L1 and L2 regularization", model_rmse(model_1i)[0], model_rmse(model_1i)[1], evaluate_1i[0], evaluate_1i[1]],
                 ["model_1j",  "Applying Dropout", model_rmse(model_1j)[0], model_rmse(model_1j)[1], evaluate_1j[0], evaluate_1j[1]],
                 ["model_1k",  "Applying BatchNormalization", model_rmse(model_1k)[0], model_rmse(model_1k)[1], evaluate_1k[0], evaluate_1k[1]]]

all_results = pd.DataFrame(model_results, columns=["model", "Improvement", "Train Score RMSE", "Test score RMSE", "Loss", "MAE"])

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
all_results[["Train Score RMSE", "Test score RMSE", "MAE"]].plot(ax=ax[0], kind='bar')
all_results[["Loss"]].plot(ax=ax[1], kind='bar')
plt.show()