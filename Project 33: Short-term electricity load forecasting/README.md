## PROJECT 33: Short-term electricity load forecasting

> ### TASK: Stady 1D Convolutional Neural Network for forecasting time series


### Project goals and objectives

#### Project goal

- Studying **1D Convolutional Neural Network for forecasting time series**

#### Project objectives

1. Explore and prepare data 
2. Building 1D CNN model

### Dataset

[Short-term electricity load forecasting](https://www.kaggle.com/ernestojaguilar/shortterm-electricity-load-forecasting-panama)

**DATASET INFORMATION:**

These datasets are framed on predicting the short-term electricity, this forecasting problem is known in the research field as short-term load forecasting (STLF). These datasets address the STLF problem for the Panama power system, in which the forecasting horizon is one week, with hourly steps, which is a total of 168 hours. These datasets are useful to train and test forecasting models and compare their results with the power system operator official forecast (take a look at real-time electricity load). The datasets include historical load, a vast set of weather variables, holidays, and historical load weekly forecast features. More information regarding these datasets context, a literature review of forecasting techniques suitable for this dataset, and results after testing a set of Machine Learning; are available in the article Short-Term Electricity Load Forecasting with Machine Learning. (Aguilar Madrid, E.; Antonio, N. Short-Term Electricity Load Forecasting with Machine Learning. Information 2021, 12, 50. https://doi.org/10.3390/info12020050)

**Datasets**

For simplicity, the published datasets are already pre-processed by merging all data sources on the date-time index:

- A CSV file containing all records in a single continuous dataset with all variables.
- A CSV file containing the load forecast from weekly pre-dispatch reports.
- Two Excel files containing suggested regressors and 14 pairs of training/testing datasets as described in the PDF file.

These 14 pairs of raining/testing datasets are selected according to these testing criteria:

- A testing week for each month before the lockdown due to COVID-19.
- Select testing weeks containing holidays.
- Plus, two testing weeks during the lockdown.


### Results

1. [ ] [**1D CNN model**]()


### References

1. [Short-Term Electricity Load Forecasting with Machine Learning](https://www.mdpi.com/2078-2489/12/2/50/htm)
2. [Temporal Coils: Intro to Temporal Convolutional Networks for Time Series Forecasting in Python](https://towardsdatascience.com/temporal-coils-intro-to-temporal-convolutional-networks-for-time-series-forecasting-in-python-5907c04febc6)
3. [1-d Convolutional Neural Networks for Time Series: Basic Intuition](https://boostedml.com/2020/04/1-d-convolutional-neural-networks-for-time-series-basic-intuition.html)
4. [How to Develop Convolutional Neural Network Models for Time Series Forecastin](https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/)
