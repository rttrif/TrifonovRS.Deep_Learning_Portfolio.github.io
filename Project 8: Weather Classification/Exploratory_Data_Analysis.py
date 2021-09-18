"""
PROJECT 8: Weather Classification
TASK: Multi-class classification
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: ResNet
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training ResNet-34
3. Training ResNet-50
4. Training ResNet-101
"""
# %%
# IMPORT LIBRARIES

# GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
import itertools

# %%
# PATH & LABEL PROCESS

# Main path
data_path = Path('data/Weather_Dataset')

# Listing subdirectories
file_path = list(data_path.glob('**/*.jpg'))

# Mapping the labels
img_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_path))

# Transformation to series
files = pd.Series(file_path, name="files", dtype='object').astype(str)
labels = pd.Series(img_labels, name="category", dtype='object')

# Checking results
print(files)
print(labels)

# Concatenating series to train_data dataframe
train_df = pd.concat([files, labels], axis=1)

# Shuffling
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Checking results
print(train_df.head(-1))
print(train_df.info())

# %%
# VISUALIZATION: Overview of images

# General
sns.countplot(train_df["category"])
plt.show()

train_df['category'].value_counts().plot.pie(figsize=(5, 5))
plt.show()

# Random example
figure = plt.figure(figsize=(10, 10))
ind = np.random.randint(0, train_df["files"].shape[0])
x = cv2.imread(train_df["files"][ind])
plt.imshow(x)
plt.xlabel(x.shape)
plt.title(train_df["category"][ind])
plt.show()

# Several  examples
fig, axes = plt.subplots(nrows=5,
                         ncols=5,
                         figsize=(10, 10),
                         subplot_kw={"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    x = cv2.imread(train_df["files"][i])
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ax.imshow(x)
    ax.set_title(train_df["category"][i])
plt.tight_layout()
plt.show()
