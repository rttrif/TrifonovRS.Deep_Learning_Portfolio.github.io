"""
PROJECT 6: Vehicle Classification

TASK: Classification

PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
> Studying architecture: AlexNet

PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training AlexNet
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
data_path = Path('data')

# Path process
img_path = list(data_path.glob(r"*/*.png"))

# Mapping the labels
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], img_path))

# Transformation to series structure
img_path_series = pd.Series(img_path, name="PICTURE").astype(str)
img_labels_series = pd.Series(img_labels, name="CATEGORY")

# Checking results
print(img_path_series)
print(img_labels_series)

# Concatenating series to train_data dataframe
train_data = pd.concat([img_path_series, img_labels_series], axis=1)

# Shuffling
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Checking results
print(train_data.head(-1))
print(train_data.info())

# %%
# VISUALIZATION: Overview of images

# General
sns.countplot(train_data["CATEGORY"])
plt.show()

train_data['CATEGORY'].value_counts().plot.pie(figsize=(5, 5))
plt.show()

# Random example
figure = plt.figure(figsize=(10, 10))
ind = np.random.randint(0, train_data["PICTURE"].shape[0])
x = cv2.imread(train_data["PICTURE"][ind])
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
    x = cv2.imread(train_data["PICTURE"][i])
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ax.imshow(x)
    ax.set_title(train_data["CATEGORY"][i])
plt.tight_layout()
plt.show()


















