"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL

Development of skills for building and improving simple CNN models for multi-class classification
using the sequential Tensorflow API.

STAGE OBJECTIVES
1. EDA
2. Data preparation
3. Training simple CNN model

DATASET: COVID-19 Radiography Database

ATTRIBUTE INFORMATION:
1. COVID data
2. Normal images
3. Lung opacity images
4. Viral Pneumonia images
"""
# %%
# IMPORT LIBRARIES

# Data Preprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from glob import glob
from PIL import Image
import os
import random
import cv2

# %%
# IMPORT DATA

# List number of files
for dir_path, dir_names, file_names in os.walk("data/COVID19_Radiography_Dataset/"):
    print(f"There are {len(dir_names)} directories and {len(file_names)} images in '{dir_path}'.")

path = '/Users/rttrif/Data_Science_Projects/Tensorflow_Certification/Prokect_4_COVID_19_Radiography_Database/' \
       'data/COVID19_Radiography_Dataset'

diag_code_dict = {
    'COVID': 0,
    'Lung_Opacity': 1,
    'Normal': 2,
    'Viral Pneumonia': 3}

diag_title_dict = {
    'COVID': 'Covid-19',
    'Lung_Opacity': 'Lung Opacity',
    'Normal': 'Healthy',
    'Viral Pneumonia': 'Viral Pneumonia'}

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*', '*.png'))}

# %%
# EXPLORATORY DATA ANALYSIS

covid_data = pd.DataFrame.from_dict(imageid_path_dict, orient='index').reset_index()
covid_data.columns = ['image_id', 'path']
classes = covid_data.image_id.str.split('-').str[0]

covid_data['diag'] = classes
covid_data['target'] = covid_data['diag'].map(diag_code_dict.get)
covid_data['Class'] = covid_data['diag'].map(diag_title_dict.get)
# %%
# Simple EDA

samples, feature = covid_data.shape
duplicated = covid_data.duplicated().sum()
null_values = covid_data.isnull().sum().sum()

print('Simple EDA')
print('Number of samples: %d' % (samples))
print('duplicates: %d' % (duplicated))
print('null values: %d' % (null_values))
# %%
# Samples per class
plt.figure(figsize=(20, 8))
sns.set(style="ticks", font_scale=1)
ax = sns.countplot(data=covid_data, x='Class', order=covid_data['Class'].value_counts().index, palette="flare")
sns.despine(top=True, right=True, left=True, bottom=False)
plt.xticks(rotation=0, fontsize=12)
ax.set_xlabel('Sample Type - Diagnosis', fontsize=14, weight='bold')
ax.set(yticklabels=[])
ax.axes.get_yaxis().set_visible(False)
plt.title('Number of Samples per Class', fontsize=16, weight='bold')

# Plot percentage
for p in ax.patches:
    ax.annotate("%.1f%%" % (100 * float(p.get_height() / samples)),
                (p.get_x() + p.get_width() / 2., abs(p.get_height())),
                ha='center', va='bottom', color='black', xytext=(0, 10), rotation='horizontal',
                textcoords='offset points')
plt.show()
# %%
# Samples per class
covid_data['image'] = covid_data['path'].map(lambda x: np.asarray(Image.open(x).resize((75, 75))))

n_samples = 3

fig, m_axs = plt.subplots(4, n_samples, figsize=(4 * n_samples, 3 * 4))

for n_axs, (type_name, type_rows) in zip(m_axs, covid_data.sort_values(['diag']).groupby('diag')):
    n_axs[1].set_title(type_name, fontsize=14, weight='bold')
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=42).iterrows()):
        picture = c_row['path']
        image = cv2.imread(picture)
        c_ax.imshow(image)
        c_ax.axis('off')
plt.show()

# %%
print('Shape of the image: {}'.format(image.shape))
print('Image size {}'.format(image.size))
image.dtype
print('Max rgb: {}'.format(image.max()))
print('Min rgb: {}'.format(image.min()))
image[0, 0]

# %%
mean_val = []
std_dev_val = []
max_val = []
min_val = []

for i in range(0, samples):
    mean_val.append(covid_data['image'][i].mean())
    std_dev_val.append(np.std(covid_data['image'][i]))
    max_val.append(covid_data['image'][i].max())
    min_val.append(covid_data['image'][i].min())

imageEDA = covid_data.loc[:, ['image', 'Class', 'path']]
imageEDA['mean'] = mean_val
imageEDA['stedev'] = std_dev_val
imageEDA['max'] = max_val
imageEDA['min'] = min_val

subt_mean_samples = imageEDA['mean'].mean() - imageEDA['mean']
imageEDA['subt_mean'] = subt_mean_samples

# %%
# VISUALIZATION

# Image color mean value distribution
sns.displot(data=imageEDA, x='mean', kind="kde")
plt.title('Image color mean value distribution')
plt.show()

# Image color mean value distribution by class
sns.displot(data=imageEDA, x='mean', kind="kde", hue='Class')
plt.title('Image color mean value distribution by class')
plt.show()

# Image color max value distribution by class
sns.displot(data=imageEDA, x='max', kind="kde", hue='Class')
plt.title('Image color max value distribution by class')
plt.show()

# Image color min value distribution by class
sns.displot(data=imageEDA, x='min', kind="kde", hue='Class')
plt.title('Image color min value distribution by class')
plt.show()

# Mean and Standard Deviation of Image Samples
plt.figure(figsize=(20, 8))
sns.set(style="ticks", font_scale=1)
ax = sns.scatterplot(data=imageEDA, x="mean", y=imageEDA['stedev'], hue='Class', alpha=0.8)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xticks(rotation=0, fontsize=12)
ax.set_xlabel('Image Channel Colour Mean', fontsize=14, weight='bold')
ax.set_ylabel('Image Channel Colour Standard Deviation', fontsize=14, weight='bold')
plt.title('Mean and Standard Deviation of Image Samples', fontsize=16, weight='bold')
plt.show()

# Mean and standard dev of img samples
plt.figure(figsize=(20,8))
g = sns.FacetGrid(imageEDA, col="Class", height=5)
g.map_dataframe(sns.scatterplot, x='mean', y='stedev')
g.set_titles(col_template="{col_name}", row_template="{row_name}", size=16)
g.fig.subplots_adjust(top=.7)
g.fig.suptitle('Mean and standard dev of img samples')
axes = g.axes.flatten()
axes[0].set_ylabel('std dev')
for ax in axes:
    ax.set_xlabel('Mean')
g.fig.tight_layout()
plt.show()
