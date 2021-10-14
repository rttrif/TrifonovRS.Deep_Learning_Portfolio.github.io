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
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_hub as hub

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
def preprocess_img(image, label, img_shape=256):
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

# %%
# FEATURE EXTRACTION MODEL

# Base model
input_shape = (256, 256, 3)
base_model = hub.KerasLayer("https://tfhub.dev/adityakane2001/regnety400mf_classification/1", include_top=False)
base_model.trainable = False
