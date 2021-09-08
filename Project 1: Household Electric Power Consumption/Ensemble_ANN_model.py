"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models
using the sequential and functional Tensorflow API

PROJECT OBJECTIVES
Training an ensemble of models
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Average
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import Regularizer
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
# model_1k
def model_1k(epochs, patience):
    RegularizedDense = partial(tf.keras.layers.Dense,
                               activation="relu",
                               kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.l1_l2(0.01))

    model = Sequential([
        RegularizedDense(128, input_shape=(6,)),
        Dropout(rate=0.3),
        BatchNormalization(),

        RegularizedDense(256),
        Dropout(rate=0.3),
        BatchNormalization(),

        RegularizedDense(128),
        Dropout(rate=0.3),
        BatchNormalization(),

        RegularizedDense(64),
        Dropout(rate=0.3),
        BatchNormalization(),

        Dense(1, activation='linear')
    ])

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()

    tf.keras.utils.plot_model(model, to_file='model_1k_ensem.png')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_1k_ensem.h5",
                                                       save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=[checkpoint_cb, early_stopping_cb])

    return history, model


history, model_1k = model_1k(epochs=500, patience=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1k = model_1k.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_1k = model_rmse(model_1k)[0]
test_score_1k = model_rmse(model_1k)[1]


# %%
# model_2h
def model_2h(epochs, patience):
    input = Input(shape=(6,), name='model_2h')

    # Deep branch 1
    branch_11 = Dense(256, activation='tanh', kernel_regularizer='l1')(input)
    dropout = Dropout(rate=0.3)(branch_11)
    bach_norm = BatchNormalization()(dropout)

    branch_12 = Dense(128, activation='tanh', kernel_regularizer='l1')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_12)
    out_branch_1 = BatchNormalization()(dropout)

    # Deep branch 2
    branch_21 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(input)
    dropout = Dropout(rate=0.3)(branch_21)
    bach_norm = BatchNormalization()(dropout)

    branch_22 = Dense(512, activation='sigmoid', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_22)
    bach_norm = BatchNormalization()(dropout)

    branch_23 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_23)
    bach_norm = BatchNormalization()(dropout)

    branch_24 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_24)
    out_branch_2 = BatchNormalization()(dropout)

    # Deep branch 3
    branch_31 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(input)
    dropout = Dropout(rate=0.3)(branch_31)
    bach_norm = BatchNormalization()(dropout)

    branch_32 = Dense(256, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_32)
    bach_norm = BatchNormalization()(dropout)

    branch_33 = Dense(512, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_33)
    bach_norm = BatchNormalization()(dropout)

    branch_34 = Dense(512, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_34)
    bach_norm = BatchNormalization()(dropout)

    branch_35 = Dense(256, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_35)
    bach_norm = BatchNormalization()(dropout)

    branch_36 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(bach_norm)
    dropout = Dropout(rate=0.3)(branch_36)
    out_branch_3 = BatchNormalization()(dropout)

    concat = Concatenate()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2h')

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2h.png')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_2h.h5",
                                                       save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=[checkpoint_cb, early_stopping_cb])

    return history, model


history, model_2h = model_2h(epochs=500, patience=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2h = model_2h.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2h = model_rmse(model_2h)[0]
test_score_2h = model_rmse(model_2h)[1]


# %%
# model_1k + model_2h

def ensemble_1k_2h(epochs, patience):
    input = Input(shape=(6,), name='ensemble_1k_2h')

    model1 = model_1k(input)
    model2 = model_2h(input)

    ensemble = Average()([model1, model2])
    output = Dense(1, activation='linear')(ensemble)

    model = Model(inputs=input, outputs=output, name='ensemble_1k_2h')

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='ensemble_1k_2h.png')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("ensemble_1k_2h.h5",
                                                       save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience,
                                                         restore_best_weights=True)

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2,
                        callbacks=[checkpoint_cb, early_stopping_cb])

    return history, model


history, ensemble_1k_2h = ensemble_1k_2h(epochs=500, patience=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_ensemble_1k_2h = ensemble_1k_2h.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_ensemble_1k_2h = model_rmse(ensemble_1k_2h)[0]
test_score_ensemble_1k_2h = model_rmse(ensemble_1k_2h)[1]
# %%
# COMPARISON OF MODEL RESULTS

# 0 - Simple ANN
# 1 - Complex ANN
# 2 - Ensemble ANN

model_results = [["model_1k", "Simple ANN", train_score_1k, test_score_1k, evaluate_1k[0], evaluate_1k[1]],
                 ["model_2h", "Complex ANN", train_score_2h, test_score_2h, evaluate_2h[0], evaluate_2h[1]],
                 ["ensemble_1k_2h", "Ensemble ANN", train_score_ensemble_1k_2h, test_score_ensemble_1k_2h, evaluate_ensemble_1k_2h[0], evaluate_ensemble_1k_2h[1]]]

all_results = pd.DataFrame(model_results,
                           columns=["model", "Type ANN", "Train Score RMSE", "Test score RMSE", "Loss", "MAE"])

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
all_results[["Train Score RMSE", "Test score RMSE", "MAE"]].plot(ax=ax[0], kind='bar')
all_results[["Loss"]].plot(ax=ax[1], kind='bar')
plt.show()
