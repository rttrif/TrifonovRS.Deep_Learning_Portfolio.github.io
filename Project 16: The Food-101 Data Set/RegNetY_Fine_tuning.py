"""
PROJECT 16: Food 101
TASKS: Classification
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
- Studying feature extraction transfer learning
- Studying Tensorflow Datasets
PROJECT OBJECTIVES
1. Using TensorFlow Datasets to download and explore data
2. Building a feature extraction model
3. Viewing training results on TensorBoard
"""
# %%
# IMPORT LIBRARIES
import tensorboard
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_hub as hub
from tensorflow.keras.applications import EfficientNetB0

import pydot
import graphviz
import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load dataset
(train_data, test_data), ds_food_101 = tfds.load(name="food101",
                                                 split=["train", "validation"],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)
# %%
# DATA EXPLORATION

# Features
ds_food_101.features

# Get class names
class_names = ds_food_101.features["label"].names
class_names[:10]

# Sample the training data
train_one_sample = train_data.take(145)

# Info about sample
for image, label in train_one_sample:
    print(f"""
  Image shape: {image.shape}
  Image dtype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
        """)

# Plot an image tensor
plt.imshow(image)
plt.title(class_names[label.numpy()])
plt.axis(False)
plt.show()

# %%
# DATA PREPARATION

# Estimation of the range of changes in values
tf.reduce_min(image), tf.reduce_max(image)


# Preprocessing images
def preprocess_img(image, label, img_shape=128):
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label


# Preprocess a single sample image and check the outputs
preprocessed_img = preprocess_img(image, label)[0]
print(f"Image before preprocessing:\n {image[:2]}...,\n"
      f"Shape: {image.shape},\n"
      f"Datatype: {image.dtype}\n")
print(f"Image after preprocessing:\n {preprocessed_img[:2]}...,\n"
      f"Shape: {preprocessed_img.shape},\n"
      f"Datatype: {preprocessed_img.dtype}")

# Plot an image tensor
plt.imshow(preprocessed_img / 255.)
plt.title(class_names[label])
plt.axis(False)
plt.show()

# %%
# Batch and prepare datasets

# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

# Check result
print(train_data)
print(test_data)

# %%
# Model callbacks
checkpoint_path = "model_checkpoints/cp.ckpt"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      montior="val_acc",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      verbose=0)


def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

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

# %%
# FEATURE EXTRACTION MODEL

# Base model
input_shape = (128, 128, 3)
base_model = hub.KerasLayer("https://tfhub.dev/adityakane2001/regnety400mf_feature_extractor/1", trainable=True)

# Feature extraction model
inputs = Input(shape=input_shape, name="input_layer")
x = base_model(inputs)
x = GlobalAveragePooling2D(name="pooling_layer")(x)
outputs = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs, outputs)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer='Adam',
              metrics=["accuracy"])

model.summary()
tf.keras.utils.plot_model(model, to_file='RegNetY_Fine_tuning.png')

# %%
# Train model
# Fit the model with callbacks
history = model.fit(train_data,
                    epochs=5,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=int(0.15 * len(test_data)),
                    callbacks=[create_tensorboard_callback("training_logs",
                                                           "efficientnetb0_101_classes_all_data_feature_extract"),
                               model_checkpoint])
# %%
# EVALUATION RESULT

# Learning curves
learning_curves(history)

# Evaluation model
evaluation_model(history)

# Evaluate the model on the test set
evaluate = model.evaluate(test_data, verbose=2)

# View training results on TensorBoard
# tensorboard dev upload --logdir ./training_logs \
#     --name "training_logs" \
#     --description "efficientnetb0_101_classes_all_data_feature_extract"
