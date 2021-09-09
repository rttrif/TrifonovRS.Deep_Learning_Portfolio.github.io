"""
PROJECT 3:  Fashion-MNIST

TASK: Multi-class classification

PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models for **multi-class  classification** using the sequential Tensorflow API

PROJECT OBJECTIVES
1. Use datasets from tf.data.datasets
2. Build and train models for multi-class categorization.
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
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.utils import to_categorical


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


def model_confusion_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# %%
# IMPORT DATA
(X_train, y_train_labels), (X_test, y_test_labels) = fashion_mnist.load_data()

# %%
# DATA EXPLORATION

# Random example
ind = np.random.randint(0, X_train.shape[0])
plt.imshow(X_train[ind])
print(f'Label is {y_train_labels[ind]}')
plt.show()

# Data shape
print(X_train.shape, X_test.shape)
print(y_train_labels.shape)

# Samples label
y_train_labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
len(class_names)

# %%
# DATA PREPARATION

# Estimation of the range of changes in values
X_train.min(), X_train.max()

# NORMALIZE DATA
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train.min(), X_train.max()

# Pulling features
X_train = X_train.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))
X_train.shape, X_test.shape

y_train = to_categorical(y_train_labels)
y_test = to_categorical(y_test_labels)

y_train[:5]


# %%
# Basic model

def model_1(epochs):
    model = Sequential([
        Dense(32, activation='sigmoid', input_shape=(28 * 28,)),
        Dense(16, activation='sigmoid'),
        Dense(8, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_1.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_1 = model_1(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1 = model_1.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_1.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)


# %%
# Basic model
# + Change the activation function in hidden layers by sigmoid to elu

def model_2(epochs):
    model = Sequential([
        Dense(32, activation='elu', input_shape=(28 * 28,)),
        Dense(16, activation='elu'),
        Dense(8, activation='elu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_2.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_2 = model_2(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_2 = model_2.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_2.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)


# %%
# Basic model
# + Change the activation function in hidden layers by elu to relu

def model_3(epochs):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(28 * 28,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_3.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_3 = model_3(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_3 = model_3.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_3.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)


# %%
# Basic model
# + Change the activation function in hidden layers by elu to relu
# + Change the optimisation function by SGD to Adam

def model_4(epochs):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(28 * 28,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_4.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_4 = model_4(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_4 = model_4.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_4.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)


# %%
# Basic model
# + Change the activation function in hidden layers by elu to relu
# + Change the optimisation function by SGD to Adam
# + Increasing the number of neurons in the hidden layers
def model_5(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(28 * 28,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_5.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_5 = model_5(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_5 = model_5.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_5.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)


# %%
# Basic model
# + Change the activation function in hidden layers by elu to relu
# + Change the optimisation function by SGD to Adam
# + Increasing the number of neurons in the hidden layers
# + Increasing the number of hidden layers
def model_6(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(28 * 28,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_6.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_6 = model_6(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_6 = model_6.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_6.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)


# %%
# Basic model
# + Change the activation function in hidden layers by elu to relu
# + Change the optimisation function by SGD to Adam
# + Increasing the number of neurons in the hidden layers
# + Increasing the number of hidden layers
# + Applying L2 regularization
def model_7(epochs):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(28 * 28,), kernel_regularizer='l2'),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dense(16, activation='relu', kernel_regularizer='l2'),
        Dense(10, activation='relu', kernel_regularizer='l2'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_7.png')

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_split=0.2)

    return history, model


history, model_7 = model_7(epochs=15)

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_7 = model_7.evaluate(X_test, y_test, verbose=2)

# Predicting the test set results
y_pred = model_7.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Confusion matrix
cnf_matrix = confusion_matrix(y_test_labels, y_pred_class)
plt.figure(figsize=(15, 15))
model_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix', normalize=False)