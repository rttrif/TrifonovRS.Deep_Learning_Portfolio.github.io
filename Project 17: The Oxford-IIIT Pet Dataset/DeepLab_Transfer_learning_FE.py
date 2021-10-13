"""
PROJECT 17: The Oxford-IIIT Pet Dataset
TASKS: Semantic Segmentation
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
- Studying **feature extraction** transfer learning
- Studying Tensorflow Datasets
Project objectives
1. Using TensorFlow Datasets to download and explore data
2. Building a feature extraction model
3. Viewing training results on TensorBoard
"""
# %%
# IMPORT LIBRARIES
import tensorflow as tf
import tensorflow_datasets as tfds

# %%
# Load dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
