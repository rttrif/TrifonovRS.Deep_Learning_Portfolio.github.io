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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
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
# Basic model
def model_2a(epochs):
    input = Input(shape=(6,), name='model_2a')

    hidden_1 = Dense(256, activation='tanh')(input)
    hidden_2 = Dense(128, activation='tanh')(hidden_1)

    concat = Concatenate()([input, hidden_2])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2a')

    model.compile(optimizer='SGD',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2a.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2a(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2a = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2a = model_rmse(model)[0]
test_score_2a = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers

def model_2b(epochs):
    input = Input(shape=(6,), name='model_2b')

    # Deep branch 1
    branch_11 = Dense(256, activation='tanh')(input)
    branch_12 = Dense(128, activation='tanh')(branch_11)

    # Deep branch 2
    branch_21 = Dense(256, activation='sigmoid')(input)
    branch_22 = Dense(512, activation='sigmoid')(branch_21)
    branch_23 = Dense(256, activation='sigmoid')(branch_22)
    branch_24 = Dense(256, activation='sigmoid')(branch_23)

    concat = Concatenate()([input, branch_12, branch_24])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2b')

    model.compile(optimizer='SGD',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2b.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2b(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2b = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2b = model_rmse(model)[0]
test_score_2b = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# + Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers

def model_2c(epochs):
    input = Input(shape=(6,), name='model_2c')

    # Deep branch 1
    branch_11 = Dense(256, activation='tanh')(input)
    branch_12 = Dense(128, activation='tanh')(branch_11)

    # Deep branch 2
    branch_21 = Dense(256, activation='sigmoid')(input)
    branch_22 = Dense(512, activation='sigmoid')(branch_21)
    branch_23 = Dense(256, activation='sigmoid')(branch_22)
    branch_24 = Dense(256, activation='sigmoid')(branch_23)

    # Deep branch 3
    branch_31 = Dense(128, activation='relu')(input)
    branch_32 = Dense(256, activation='relu')(branch_31)
    branch_33 = Dense(512, activation='relu')(branch_32)
    branch_34 = Dense(512, activation='relu')(branch_33)
    branch_35 = Dense(256, activation='relu')(branch_34)
    branch_36 = Dense(128, activation='relu')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2c')

    model.compile(optimizer='SGD',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2c.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2c(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2c = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2c = model_rmse(model)[0]
test_score_2c = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# + Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers
# + Changing the optimization function

def model_2d(epochs):
    input = Input(shape=(6,), name='model_2d')

    # Deep branch 1
    branch_11 = Dense(256, activation='tanh')(input)
    branch_12 = Dense(128, activation='tanh')(branch_11)

    # Deep branch 2
    branch_21 = Dense(256, activation='sigmoid')(input)
    branch_22 = Dense(512, activation='sigmoid')(branch_21)
    branch_23 = Dense(256, activation='sigmoid')(branch_22)
    branch_24 = Dense(256, activation='sigmoid')(branch_23)

    # Deep branch 3
    branch_31 = Dense(128, activation='relu')(input)
    branch_32 = Dense(256, activation='relu')(branch_31)
    branch_33 = Dense(512, activation='relu')(branch_32)
    branch_34 = Dense(512, activation='relu')(branch_33)
    branch_35 = Dense(256, activation='relu')(branch_34)
    branch_36 = Dense(128, activation='relu')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2d')

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2d.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2d(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2d = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2d = model_rmse(model)[0]
test_score_2d = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# + Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers
# + Changing the optimization function
# + Applying L1, L2 and L1_L2 regularization

def model_2e(epochs):
    input = Input(shape=(6,), name='model_2e')

    # Deep branch 1
    branch_11 = Dense(256, activation='tanh', kernel_regularizer='l1')(input)
    branch_12 = Dense(128, activation='tanh', kernel_regularizer='l1')(branch_11)

    # Deep branch 2
    branch_21 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(input)
    branch_22 = Dense(512, activation='sigmoid', kernel_regularizer='l2')(branch_21)
    branch_23 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(branch_22)
    branch_24 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(branch_23)

    # Deep branch 3
    branch_31 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(input)
    branch_32 = Dense(256, activation='relu', kernel_regularizer='l1_l2')(branch_31)
    branch_33 = Dense(512, activation='relu', kernel_regularizer='l1_l2')(branch_32)
    branch_34 = Dense(512, activation='relu', kernel_regularizer='l1_l2')(branch_33)
    branch_35 = Dense(256, activation='relu', kernel_regularizer='l1_l2')(branch_34)
    branch_36 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(branch_35)

    concat = Concatenate()([input, branch_12, branch_24, branch_36])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2e')

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2e.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2e(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2e = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2e = model_rmse(model)[0]
test_score_2e = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# + Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers
# + Changing the optimization function
# + Applying L1, L2 and L1_L2 regularization
# + Applying Dropout

def model_2f(epochs):
    input = Input(shape=(6,), name='model_2f')

    # Deep branch 1
    branch_11 = Dense(256, activation='tanh', kernel_regularizer='l1')(input)
    dropout = Dropout(rate=0.25)(branch_11)

    branch_12 = Dense(128, activation='tanh', kernel_regularizer='l1')(dropout)
    out_branch_1 = Dropout(rate=0.25)(branch_12)

    # Deep branch 2
    branch_21 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(input)
    dropout = Dropout(rate=0.25)(branch_21)

    branch_22 = Dense(512, activation='sigmoid', kernel_regularizer='l2')(dropout)
    dropout = Dropout(rate=0.25)(branch_22)

    branch_23 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(dropout)
    dropout = Dropout(rate=0.25)(branch_23)

    branch_24 = Dense(256, activation='sigmoid', kernel_regularizer='l2')(dropout)
    out_branch_2 = Dropout(rate=0.25)(branch_24)

    # Deep branch 3
    branch_31 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(input)
    dropout = Dropout(rate=0.25)(branch_31)

    branch_32 = Dense(256, activation='relu', kernel_regularizer='l1_l2')(dropout)
    dropout = Dropout(rate=0.25)(branch_32)

    branch_33 = Dense(512, activation='relu', kernel_regularizer='l1_l2')(dropout)
    dropout = Dropout(rate=0.25)(branch_33)

    branch_34 = Dense(512, activation='relu', kernel_regularizer='l1_l2')(dropout)
    dropout = Dropout(rate=0.25)(branch_34)

    branch_35 = Dense(256, activation='relu', kernel_regularizer='l1_l2')(dropout)
    dropout = Dropout(rate=0.25)(branch_35)

    branch_36 = Dense(128, activation='relu', kernel_regularizer='l1_l2')(dropout)
    out_branch_3 = Dropout(rate=0.25)(branch_36)

    concat = Concatenate()([input, out_branch_1, out_branch_2, out_branch_3])
    output = Dense(1, activation='linear')(concat)

    model = Model(inputs=input, outputs=output, name='model_2f')

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2f.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2f(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2f = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2f = model_rmse(model)[0]
test_score_2f = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# + Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers
# + Changing the optimization function
# + Applying L1, L2 and L1_L2 regularization
# + Applying Dropout
# + Applying BatchNormalization

def model_2g(epochs):
    input = Input(shape=(6,), name='model_2g')

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

    model = Model(inputs=input, outputs=output, name='model_2g')

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae'])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2g.png')

    history = model.fit(X_train_norm, y_train,
                        epochs=epochs,
                        batch_size=128,
                        verbose=2,
                        validation_split=0.2)

    return history, model


history, model = model_2g(epochs=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2g = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2g = model_rmse(model)[0]
test_score_2g = model_rmse(model)[1]


# %%
# Basic model
# + Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# + Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers
# + Changing the optimization function
# + Applying L1, L2 and L1_L2 regularization
# + Applying Dropout
# + Applying BatchNormalization
# + Applying an early stopping + increasing the number of training epochs

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


history, model = model_2h(epochs=100, patience=10)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2h = model.evaluate(X_test, y_test, verbose=2)

# RMSE
train_score_2h = model_rmse(model)[0]
test_score_2h = model_rmse(model)[1]
# %%
# COMPARISON OF MODEL RESULTS

# 0 - Basic model
# 1 - Add deep branch with 4 layers + another activation function (sigmoid) + increasing the number of neurons in the hidden layers
# 2 - Add deep branch with 6 layers + another activation function (relu) + increasing the number of neurons in the hidden layers
# 3 - Changing the optimization function
# 4 - Applying L1, L2 and L1_L2 regularization
# 5 - Applying Dropout
# 6 - Applying BatchNormalization
# 7 - Applying an early stopping + increasing the number of training epochs

model_results = [["model_2a", "Basic model", train_score_2a, test_score_2a, evaluate_2a[0], evaluate_2a[1]],
                 ["model_2b",  "Branch with 4 layers + sigmoid", train_score_2b, test_score_2b, evaluate_2b[0], evaluate_2b[1]],
                 ["model_2c",  "Branch with 6 layers + relu", train_score_2c, test_score_2c, evaluate_2c[0], evaluate_2c[1]],
                 ["model_2d",  "Adam", train_score_2d, test_score_2d, evaluate_2d[0], evaluate_2d[1]],
                 ["model_2e",  "L1, L2 and L1_L2 regularization",  train_score_2e, test_score_2e, evaluate_2e[0], evaluate_2e[1]],
                 ["model_2f",  "Dropout",  train_score_2f, test_score_2f, evaluate_2f[0], evaluate_2f[1]],
                 ["model_2g",  "BatchNormalization", train_score_2g, test_score_2g, evaluate_2g[0], evaluate_2g[1]],
                 ["model_2h",  "early stopping + more epochs", train_score_2h, test_score_2h, evaluate_2h[0], evaluate_2h[1]]]

all_results = pd.DataFrame(model_results, columns=["model", "Improvement", "Train Score RMSE", "Test score RMSE", "Loss", "MAE"])

fig, ax = plt.subplots(ncols=2, figsize=(20, 8))
all_results[["Train Score RMSE", "Test score RMSE", "MAE"]].plot(ax=ax[0], kind='bar')
all_results[["Loss"]].plot(ax=ax[1], kind='bar')
plt.show()