"""
PROJECT 5: Forest Fire Detection

TASK: Classification

PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
> Studying architecture: LeNet-5

PROJECT OBJECTIVES

1. Exploratory Data Analysis
2. Training LeNet-5
"""
# %%
# IMPORT LIBRARIES

# GENERAL
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# PATH PROCESS
import os
import os.path
import glob
from pathlib import Path

# %%
# PATH & LABEL PROCESS

# Main path
data_path = Path('data/fire_dataset')

# Path process
img_path = list(data_path.glob(r"*/*.png"))

# Label process
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path))

print("FIRE: ", img_labels.count("fire_images"))
print("NO_FIRE: ", img_labels.count("non_fire_images"))

# %%
# TRANSFORMATION TO SERIES STRUCTURE
img_path_series = pd.Series(img_path, name="PNG").astype(str)
img_labels_series = pd.Series(img_labels, name="CATEGORY")

print(img_path_series)
print(img_labels_series)

img_labels_series.replace({"non_fire_images": "NO_FIRE", "fire_images": "FIRE"}, inplace=True)

print(img_labels_series)

# %%
# TRANSFORMATION TO DATAFRAME STRUCTURE

train_data = pd.concat([img_path_series, img_labels_series], axis=1)

print(train_data.head(-1))

print(train_data.info())

# %%
# Shuffling
train_data = train_data.sample(frac=1).reset_index(drop=True)
print(train_data.head(-1))

# %%
# VISUALIZATION

# General
sns.countplot(train_data["CATEGORY"])
plt.show()

train_data['CATEGORY'].value_counts().plot.pie(figsize=(5, 5))
plt.show()

# Random example
figure = plt.figure(figsize=(10, 10))
ind = np.random.randint(0, train_data["PNG"].shape[0])
x = cv2.imread(train_data["PNG"][ind])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(train_data["CATEGORY"][ind])
plt.show()

# Several  examples
fig, axes = plt.subplots(nrows=5,
                         ncols=5,
                         figsize=(10, 10),
                         subplot_kw={"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    x = cv2.imread(train_data["PNG"][i])
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ax.imshow(x)
    ax.set_title(train_data["CATEGORY"][i])
plt.tight_layout()
plt.show()
