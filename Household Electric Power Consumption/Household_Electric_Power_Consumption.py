'''
HOUSEHOLD ELECTRIC POWER CONSUMPTION

Solving the regression problem using fully connected artificial neural networks.
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# READ DATA
data_path = "Household Electric Power Consumption/household_power_consumption.txt"

data_frame = pd.read_csv(data_path, sep=';')

head = data_frame.head().T

tail = data_frame.tail().T
# %%
# EDA




