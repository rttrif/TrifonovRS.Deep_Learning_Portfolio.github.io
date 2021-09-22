"""
PROJECT 9: Brain Hemorrhage
TASK: Multi-class classification
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: Inception
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training Inception V1
3. Training Inception V2
4. Training Inception V3
5. Training Inception V4
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
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, \
     Concatenate, AvgPool2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# %%
# PATH & LABEL PROCESS

# Main path
data_path = Path('data')

# Listing subdirectories
file_path = list(data_path.glob('**/*.jpg'))

# Mapping the labels
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_path))

# Transformation to series
files = pd.Series(file_path, name="files", dtype='object').astype(str)
labels = pd.Series(img_labels, name="category", dtype='object')

# Concatenating series to train_data dataframe
train_df = pd.concat([files, labels], axis=1)

# Number of categories
print(train_df['category'].value_counts())

# Replacing categories
train_df['category'].replace(
    {"11[11]": "Hemorrhage", "11[11]": "Hemorrhage", "12[12]": "Hemorrhage", "13[13]": "Hemorrhage",
     "14[14]": "Hemorrhage", "15[15]": "Hemorrhage", "17[17]__": "Hemorrhage",
     "19[19]": "Hemorrhage", "1[1]": "Hemorrhage", "20[20]_2": "Hemorrhage",
     "21[21] _2": "Hemorrhage", "2[2]": "Hemorrhage", "3[3]": "Hemorrhage", "4[4]": "Hemorrhage", "5[5]": "Hemorrhage",
     "6[6]": "Hemorrhage", "7[7]": "Hemorrhage", "8[8]": "Hemorrhage", "9[9]": "Hemorrhage"}, inplace=True)

train_df['category'].replace(
    {"N10[N10]": "Normal", "N11[N11]": "Normal", "N12[N12]": "Normal", "N13[N13]": "Normal", "N14[N14]": "Normal",
     "N15[N15]": "Normal", "N15[N15]": "Normal",
     "N16[N16]": "Normal", "N17[N17]": "Normal", "N18[N18]": "Normal",
     "N19[N19]": "Normal", "N1[N1]": "Normal", "N20[N20]": "Normal", "N21[N21]": "Normal",
     "N22[N22]": "Normal", "N23[N23]": "Normal", "N24[N24]": "Normal",
     "N25[N25]": "Normal", "N26[N26]": "Normal", "N27[N27]": "Normal", "N2[N2]": "Normal",
     "N3[N3]": "Normal", "N4[N4]": "Normal", "N5[N5]": "Normal",
     "N6[N6]": "Normal", "N7[N7]": "Normal", "N8[N8]": "Normal", "N9[N9]": "Normal"}, inplace=True)

# Shuffling
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Checking results
print(train_df.head())
print(train_df['category'].value_counts())
print(train_df.info())

# %%
# DATA PREPARATION

# Splitting train and test
train_data, test_data = train_test_split(train_df, train_size=0.85, random_state=42)

print("Train shape: ", train_data.shape)
print("Test shape: ", test_data.shape)

print(train_data["category"].value_counts())
print(test_data["category"].value_counts())

# Converting the label to a numeric format
test_images = LabelEncoder().fit_transform(test_data["category"])

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
                                     validation_split=0.15)

test_generator = ImageDataGenerator(rescale=1. / 255)
# %%
# Applying generator and transformation to tensor
print("Preparing the training data:")
train_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="files",
                                                   y_col="category",
                                                   target_size=(256, 256),
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="training")

print("Preparing the validation data:")
valid_images = train_generator.flow_from_dataframe(dataframe=train_data,
                                                   x_col="files",
                                                   y_col="category",
                                                   target_size=(256, 256),
                                                   color_mode="rgb",
                                                   class_mode="categorical",
                                                   batch_size=32,
                                                   subset="validation")
print("Preparing the test data:")
test_images = test_generator.flow_from_dataframe(dataframe=test_data,
                                                 x_col="files",
                                                 y_col="category",
                                                 target_size=(256, 256),
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 batch_size=32)
# %%
# Checking
print("Checking the training data:")
for data_batch, label_batch in train_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

print("Checking the validation data:")
for data_batch, label_batch in valid_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break

print("Checking the test data:")
for data_batch, label_batch in test_images:
    print("DATA SHAPE: ", data_batch.shape)
    print("LABEL SHAPE: ", label_batch.shape)
    break


# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

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
# MODEL: Inception V1

# Inception module
def inception_module(x, filters):
    t1 = Conv2D(filters=filters[0], kernel_size=1, activation='relu')(x)

    t2 = Conv2D(filters=filters[1], kernel_size=1, activation='relu')(x)
    t2 = Conv2D(filters=filters[2], kernel_size=3, padding='same', activation='relu')(t2)

    t3 = Conv2D(filters=filters[3], kernel_size=1, activation='relu')(x)
    t3 = Conv2D(filters=filters[4], kernel_size=5, padding='same', activation='relu')(t3)

    t4 = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    t4 = Conv2D(filters=filters[2], kernel_size=1, activation='relu')(t4)

    output = Concatenate()([t1, t2, t3, t4])
    return output


# %%
# Model Inception V1
"""
For convolution and max pool:
TYPE: PATCH SIZE/ STRIDE
convolution: 7×7/2
==============================
For inception module: 
TYPE: NUMBERS OF FILTERS
inception(3a):  64 96 128	16	32	32
"""
input = Input(shape=(256, 256, 3))

# convolution: 7×7/2
x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu')(input)
# max pool: 3×3/2
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

# convolution: 3×3/1
x = Conv2D(filters=64, kernel_size=1, activation='relu')(x)
x = Conv2D(filters=192, kernel_size=3, padding='same', activation='relu')(x)
# max pool: 3×3/2
x = MaxPool2D(pool_size=3, strides=2)(x)

# inception (3a): 64	96	128	16	32	32
x = inception_module(x, filters=[64, 96, 128, 16, 32, 32])
# inception (3b): 128	128	192	32	96	64
x = inception_module(x, filters=[128, 128, 192, 32, 96, 64])
# max pool: 3×3/2
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

# inception (4a): 192	96	208	16	48	64
x = inception_module(x, filters=[192, 96, 208, 16, 48, 64])
# inception (4b): 160	112	224	24	64	64
x = inception_module(x, filters=[160, 112, 224, 24, 64, 64])
# inception (4c): 128	128	256	24	64	64
x = inception_module(x, filters=[128, 128, 256, 24, 64, 64])
# inception (4d): 112	144	288	32	64	64
x = inception_module(x, filters=[112, 144, 288, 32, 64, 64])
# inception (4e): 256	160	320	32	128	128
x = inception_module(x, filters=[256, 160, 320, 32, 128, 128])
# max pool: 3×3/2
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

# inception (5a): 256	160	320	32	128	128
x = inception_module(x, filters=[256, 160, 320, 32, 128, 128])
# inception (5b): 384	192	384	48	128	128
x = inception_module(x, filters=[384, 192, 384, 48, 128, 128])

# avg pool: 7×7/1
x = AvgPool2D(pool_size=7, strides=1)(x)
# dropout (40%)
x = Dropout(rate=0.4)(x)
# linear + softmax
x = Flatten()(x)
output = Dense(units=2, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])

model.summary()
tf.keras.utils.plot_model(model, to_file='Inception_V1.png')
# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

history = model.fit(train_images,
                    validation_data=valid_images,
                    epochs=5,
                    callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate_1 = model.evaluate(test_images, verbose=2)

# Predicting the test set results
y_pred = model.predict(test_images)
y_pred_class = np.argmax(y_pred, axis=1)

prediction_class = LabelEncoder().fit_transform(test_data["category"])

# Confusion matrix
model_confusion_matrix(prediction_class, y_pred_class)
