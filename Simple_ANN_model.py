"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models
using the sequential and functional Tensorflow API

PROJECT OBJECTIVES
Training a basic simple model using the sequential Tensorflow API and improving it using various methods.
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

model_1a = Sequential([
    Dense(64, activation='sigmoid', input_shape=(6,)),
    Dense(64, activation='sigmoid'),
    Dense(1, activation='linear')
])

model_1a.compile(optimizer='SGD',
                 loss='mse',
                 metrics=['mae'])

model_1a.summary()

tf.keras.utils.plot_model(model_1a, to_file='model_1a.png')

history_1a = model_1a.fit(X_train_norm, y_train,
                          epochs=10,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.2)
# Learning curves
learning_curves(history_1a)

# Evaluation model
evaluation_model(history_1a)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4

model_1b = Sequential([
    Dense(64, activation='sigmoid', input_shape=(6,)),
    Dense(64, activation='sigmoid'),
    Dense(1, activation='linear')
])

model_1b.compile(optimizer='SGD',
                 loss='mse',
                 metrics=['mae'])

model_1b.summary()

tf.keras.utils.plot_model(model_1b, to_file='model_1b.png')

history_1b = model_1b.fit(X_train_norm, y_train,
                          epochs=10,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1b)

# Evaluation model
evaluation_model(history_1b)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers

model_1c = Sequential([
    Dense(64, activation='sigmoid', input_shape=(6,)),
    Dense(64, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(1, activation='linear')
])

model_1c.compile(optimizer='SGD',
                 loss='mse',
                 metrics=['mae'])

model_1c.summary()

tf.keras.utils.plot_model(model_1c, to_file='model_1c.png')

history_1c = model_1c.fit(X_train_norm, y_train,
                          epochs=10,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1c)

# Evaluation model
evaluation_model(history_1c)

# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers

model_1d = Sequential([
    Dense(128, activation='sigmoid', input_shape=(6,)),
    Dense(256, activation='sigmoid'),
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(1, activation='linear')
])

model_1d.compile(optimizer='SGD',
                 loss='mse',
                 metrics=['mae'])

model_1d.summary()

tf.keras.utils.plot_model(model_1d, to_file='model_1d.png')

history_1d = model_1d.fit(X_train_norm, y_train,
                          epochs=10,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1d)

# Evaluation model
evaluation_model(history_1d)

# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function

model_1e = Sequential([
    Dense(128, activation='relu', input_shape=(6,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model_1e.compile(optimizer='SGD',
                 loss='mse',
                 metrics=['mae'])

model_1e.summary()

tf.keras.utils.plot_model(model_1e, to_file='model_1e.png')

history_1e = model_1e.fit(X_train_norm, y_train,
                          epochs=10,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1e)

# Evaluation model
evaluation_model(history_1e)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function
# + Changing the optimization function

model_1f = Sequential([
    Dense(128, activation='relu', input_shape=(6,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model_1f.compile(optimizer='Adam',
                 loss='mse',
                 metrics=['mae'])

model_1f.summary()

tf.keras.utils.plot_model(model_1f, to_file='model_1f.png')

history_1f = model_1f.fit(X_train_norm, y_train,
                          epochs=10,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1f)

# Evaluation model
evaluation_model(history_1f)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function
# + Changing the optimization function
# + Increasing the number of training epochs

model_1g = Sequential([
    Dense(128, activation='relu', input_shape=(6,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model_1g.compile(optimizer='Adam',
                 loss='mse',
                 metrics=['mae'])

model_1g.summary()

tf.keras.utils.plot_model(model_1g, to_file='model_1g.png')

history_1g = model_1g.fit(X_train_norm, y_train,
                          epochs=50,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1g)

# Evaluation model
evaluation_model(history_1g)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function
# + Changing the optimization function
# + Increasing the number of training epochs
# + Applying an early stopping and saving the best model

model_1h = Sequential([
    Dense(128, activation='relu', input_shape=(6,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model_1h.compile(optimizer='Adam',
                 loss='mse',
                 metrics=['mae'])

model_1h.summary()

tf.keras.utils.plot_model(model_1h, to_file='model_1h.png')

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_1h.h5",
                                                   save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

history_1h = model_1h.fit(X_train_norm, y_train,
                          epochs=50,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1h)

# Evaluation model
evaluation_model(history_1h)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function
# + Changing the optimization function
# + Increasing the number of training epochs
# + Applying an early stopping and saving the best model
# + Applying L1 and L2 regularization

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l1_l2(0.01))

model_1i = Sequential([
    RegularizedDense(128, input_shape=(6,)),
    RegularizedDense(256),
    RegularizedDense(128),
    RegularizedDense(64),
    Dense(1, activation='linear')
])

model_1i.compile(optimizer='Adam',
                 loss='mse',
                 metrics=['mae'])

model_1i.summary()

tf.keras.utils.plot_model(model_1i, to_file='model_1i.png')

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_1i.h5",
                                                   save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3,
                                                     restore_best_weights=True)

history_1i = model_1i.fit(X_train_norm, y_train,
                          epochs=50,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1i)

# Evaluation model
evaluation_model(history_1i)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function
# + Changing the optimization function
# + Increasing the number of training epochs
# + Applying an early stopping and saving the best model
# + Applying L1 and L2 regularization
# + Applying Dropout

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l1_l2(0.01))

model_1j = Sequential([
    RegularizedDense(128, input_shape=(6,)),
    Dropout(rate=0.3),
    RegularizedDense(256),
    Dropout(rate=0.3),
    RegularizedDense(128),
    Dropout(rate=0.3),
    RegularizedDense(64),
    Dropout(rate=0.3),
    Dense(1, activation='linear')
])

model_1j.compile(optimizer='Adam',
                 loss='mse',
                 metrics=['mae'])

model_1j.summary()

tf.keras.utils.plot_model(model_1j, to_file='model_1j.png')

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_1j.h5",
                                                   save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3,
                                                     restore_best_weights=True)

history_1j = model_1j.fit(X_train_norm, y_train,
                          epochs=50,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1j)

# Evaluation model
evaluation_model(history_1j)
# %%
# Basic model with three fully connected layers
# + Increasing validation_split to 0.4
# + Adding two additional layers
# + Increasing the number of neurons in the hidden layers
# + Changing the activation function
# + Changing the optimization function
# + Increasing the number of training epochs
# + Applying an early stopping and saving the best model
# + Applying L1 and L2 regularization
# + Applying Dropout
# + Applying BatchNormalization

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l1_l2(0.01))

model_1k = Sequential([
    RegularizedDense(128, input_shape=(6,)),
    BatchNormalization(),
    Dropout(rate=0.3),

    RegularizedDense(256),
    BatchNormalization(),
    Dropout(rate=0.3),

    RegularizedDense(128),
    BatchNormalization(),
    Dropout(rate=0.3),

    RegularizedDense(64),
    BatchNormalization(),
    Dropout(rate=0.3),

    Dense(1, activation='linear')
])

model_1k.compile(optimizer='Adam',
                 loss='mse',
                 metrics=['mae'])

model_1k.summary()

tf.keras.utils.plot_model(model_1k, to_file='model_1k.png')

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_1k.h5",
                                                   save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3,
                                                     restore_best_weights=True)

history_1k = model_1k.fit(X_train_norm, y_train,
                          epochs=50,
                          batch_size=128,
                          verbose=2,
                          validation_split=0.4)
# Learning curves
learning_curves(history_1k)

# Evaluation model
evaluation_model(history_1k)


# %%
# COMPARISON OF MODEL RESULTS

def model_rmse(model):
    # Predictions
    y_preds_train = model.predict(X_train)
    y_preds_test = model.predict(X_test)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train, y_preds_train))
    # print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(y_test, y_preds_test))
    # print('Test Score: %.2f RMSE' % (testScore))
    return trainScore, testScore


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