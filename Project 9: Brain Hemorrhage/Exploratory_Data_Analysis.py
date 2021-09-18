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
data_path = Path('data')

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
