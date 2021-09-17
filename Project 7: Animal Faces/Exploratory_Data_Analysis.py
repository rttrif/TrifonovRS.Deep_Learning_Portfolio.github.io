"""
PROJECT 7: Vehicle Classification
TASK: Multi-class classification
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
> Studying architecture: VGG
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training VGG-16
3. Training VGG-19
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
data_path = 'data/afhq/train/'

classes = {"cat": "0", "dog": "1", "wild": "2"}

all_filenames_train = []
all_categories_train = []
for class_element in classes:
    filenames = os.listdir(data_path + class_element)
    all_filenames_train += [class_element + "/" + file for file in filenames]
    all_categories_train += [classes[class_element]] * len(filenames)

train_df = pd.DataFrame({'PICTURE': all_filenames_train,
                         'CATEGORY': all_categories_train})

# Checking results
print(train_df.tail())
print(train_df.info())

all_filenames_test = []
all_categories_test = []
for class_element in classes:
    filenames = os.listdir(data_path + class_element)
    all_filenames_test += [class_element + "/" + file for file in filenames]
    all_categories_test += [classes[class_element]] * len(filenames)

test_df = pd.DataFrame({'PICTURE': all_filenames_test,
                        'CATEGORY': all_categories_test})

# Checking results
print(test_df.tail())
print(test_df.info())

# %%
# VISUALIZATION: Distribution of classes

# General
sns.countplot(train_df["CATEGORY"])
plt.title('Distribution of classes in train dataframe')
plt.show()

plt.title('Distribution of classes in train dataframe')
train_df['CATEGORY'].value_counts().plot.pie(figsize=(5, 5))
plt.show()

sns.countplot(test_df["CATEGORY"])
plt.title('Distribution of classes in test dataframe')
plt.show()

plt.title('Distribution of classes in test dataframe')
test_df['CATEGORY'].value_counts().plot.pie(figsize=(5, 5))
plt.show()


