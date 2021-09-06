"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models
using the sequential and functional Tensorflow API

PROJECT OBJECTIVES
1. Data preparation
2. Training a basic simple model using the sequential Tensorflow API and improving it using various methods.
3. Training the basic complex model using the Tensorflow functional API and improving it using various methods.
4. Training an ensemble of several models

DATASET - Household Electric Power Consumption

ATTRIBUTE INFORMATION:

1.date: Date in format dd/mm/yyyy

2.time: time in format hh:mm:ss

3.globalactivepower: household global minute-averaged active power (in kilowatt)

4.globalreactivepower: household global minute-averaged reactive power (in kilowatt)

5.voltage: minute-averaged voltage (in volt)

6.global_intensity: household global minute-averaged current intensity (in ampere)

7.submetering1: energy sub-metering No. 1 (in watt-hour of active energy).
It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric
but gas powered).

8.submetering2: energy sub-metering No. 2 (in watt-hour of active energy).
It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.

9.submetering3: energy sub-metering No. 3 (in watt-hour of active energy).
It corresponds to an electric water-heater and an air-conditioner.
"""
# %%
# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# READ DATA

data_path = "/Users/rttrif/Data_Science_Projects/Tensorflow_Certification/" \
            "Prokect_1_Household_Electric_Power_Consumption" \
            "/data/household_power_consumption.txt"

data = pd.read_csv(data_path, sep=';',
                   parse_dates={'data': ['Date', 'Time']},
                   infer_datetime_format=True,
                   na_values=['nan', '?'],
                   index_col='data')

data_head = data.head()

data_tail = data.tail()
# %%
# DATA EXPLORATION

descriptive_statistics = data.describe()

info = data.info()

missing_values = data.isna().sum()

data_clear = data.dropna()
# %%
# VISUALIZATION

# Scatter plot of Global_active_power
plt.figure(figsize=(20, 5))
plt.scatter(data_clear.index, data_clear['Global_active_power'])
plt.show()

# Global_active_power resampled over day for sum
plt.figure(figsize=(20, 5))
data_clear.Global_active_power.resample('D').sum().plot(title='Global_active_power resampled over day for sum')
plt.show()

# Global_active_power resampled over day for mean
plt.figure(figsize=(20, 5))
data_clear.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day for mean')
plt.show()

# Global_active_power resampled over day for median
plt.figure(figsize=(20, 5))
data_clear.Global_active_power.resample('D').median().plot(title='Global_active_power resampled over day for median')
plt.show()

# Global active power per hour
h = data_clear.Global_active_power.resample('h').agg(['mean', 'std', 'max', 'min'])
h.plot(subplots=True, figsize=(20, 10), title='Global active power per hour')
plt.show()
# %%
# DATA PREPARATION

# Resampling of data over 30 minutes
data_resample = data_clear.resample('30Min').mean()
data_resample.shape

# Splitting into target variable and feathers
X = data_resample.drop("Global_active_power", axis=1)
X = X.reset_index(drop=True)
X.head()

y = data_resample["Global_active_power"].reset_index(drop=True)
y.head()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Control missing values
X_train.isna().sum()

X_test.isna().sum()

y_train.isna().sum()

y_test.isna().sum()

# Drop missing values
X_train = X_train.dropna()

X_test = X_test.dropna()

y_train = y_train.dropna()

y_test = y_test.dropna()

#%%
# Save splited data

# X_train
pd.DataFrame.to_csv(X_train, 'data/X_train.csv', index=False)

# X_test
pd.DataFrame.to_csv(X_test, 'data/X_test.csv', index=False)

# y_train
pd.DataFrame.to_csv(y_train, 'data/y_train.csv', index=False)

# y_test
pd.DataFrame.to_csv(y_test, 'data/y_test.csv', index=False)