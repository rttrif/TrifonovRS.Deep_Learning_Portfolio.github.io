'''
HOUSEHOLD ELECTRIC POWER CONSUMPTION

Solving the regression problem using fully connected artificial neural networks.
'''
# IMPORT LIBRARIES
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# %%
# READ DATA
data_path = "Household Electric Power Consumption/household_power_consumption.txt"

data = pd.read_csv(data_path, sep=';',
                   parse_dates={'data' : ['Date', 'Time']},
                   infer_datetime_format=True,
                   na_values=['nan','?'],
                   index_col='data')

head = data.head().T

tail = data.tail().T
# %%
# DATA EXPLARATION

statistic_data = data.describe()

info_data = data.info()

na_data = data.isna().sum()

# Drop the null values

data = data.dropna()

data.isna().sum()
# %%
# Scatter plot of Global_active_power
plt.figure(figsize=(10,10))
plt.scatter(data.index, data['Global_active_power'])
plt.show()
#%%
# Global_active_power resampled over day for sum
plt.figure(figsize=(20,10))
data.Global_active_power.resample('D').sum().plot(title = 'Global_active_power resampled over day for sum')
plt.show()
#%%
# Global_active_power resampled over day for mean
plt.figure(figsize=(20,10))
data.Global_active_power.resample('D').mean().plot(title = 'Global_active_power resampled over day for mean')
plt.show()
#%%
# Global_active_power resampled over day for median
plt.figure(figsize=(20,10))
data.Global_active_power.resample('D').median().plot(title = 'Global_active_power resampled over day for median')
plt.show()
#%%
# Global active power per hour
h = data.Global_active_power.resample('h').agg(['mean', 'std', 'max', 'min'])
h.plot(subplots = True, figsize=(20,10), title = 'Global active power per hour')
plt.show()
#%%
# DATA PREPARETION
# Resampling of data over 30 minutes
data_resample = data.resample('30Min').mean()
data_resample.shape

#%%
# Splitting into target variable and feachers
X = data_resample.drop("Global_active_power", axis=1)
y = data_resample["Global_active_power"]

X.head().T
# y.head()
#%%
# Normalize data
X_scaled = MinMaxScaler().fit(X).transform(X)
print(X_scaled)
#%%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#%%
'''
EXPERIMENTS WITH PARAMETERS AND ARCHITECTURES OF NEURAL NETWORKS

Experement 1: Different number of layers with the same number of neurons

Experement 2: Different number of layers and different number of neurons

Experement 3: Various optimizers

Experement 4: Various activation functions

Experement 5: Complex architectures
'''
# EXPEREMENT 1: Different number of layers with the same number of neurons