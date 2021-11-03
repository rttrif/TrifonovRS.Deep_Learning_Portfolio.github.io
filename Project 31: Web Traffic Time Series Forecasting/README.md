## PROJECT 31: Web Traffic Time Series Forecasting

> ### TASK: Stady simple forecasting method 


### Project goals and objectives

#### Project goal

- Studying **Mean method**
- Studying **Naïve method**
- Studying **Seasonal naïve method**
- Studying **Drift method**

#### Project objectives

1. Explore and prepare data 
2. Building different simple models 

### Dataset

[Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data)

**DATASET INFORMATION:**

The training dataset consists of approximately 145k time series. Each of these time series represent a number of daily views of a different Wikipedia article, starting from July, 1st, 2015 up until December 31st, 2016. 


For each time series, you are provided the name of the article as well as the type of traffic that this time series represent (all, mobile, desktop, spider). You may use this metadata and any other publicly available data to make predictions. Unfortunately, the data source for this dataset does not distinguish between traffic values of zero and missing values. A missing value may mean the traffic was zero or that the data is not available for that day.

To reduce the submission file size, each page and date combination has been given a shorter Id. The mapping between page names and the submission Id is given in the key files.

**File descriptions**

Files used for the first stage will end in '1'. Files used for the second stage will end in '2'. Both will have identical formats. The complete training data for the second stage will be made available prior to the second stage.

`**train_**.csv` - contains traffic data. This a csv file where each row corresponds to a particular article and each column correspond to a particular date. Some entries are missing data. The page names contain the Wikipedia project (e.g. en.wikipedia.org), type of access (e.g. desktop) and type of agent (e.g. spider). In other words, each article name has the following format: 'name_project_access_agent' (e.g. 'AKB48_zh.wikipedia.org_all-access_spider').

`**key_**.csv` - gives the mapping between the page names and the shortened Id column used for prediction

`**sample_submission_**.csv` - a submission file showing the correct format


### Results

1. [ ] [**Mean method**]()
2. [ ] [**Naïve method**]()
3. [ ] [**Seasonal naïve method**]()
4. [ ] [**Drift method**]()

### References

1. [Some simple forecasting methods](https://otexts.com/fpp3/simple-methods.html#na%C3%AFve-method)
2. [Forecasting Methods : Part I](https://medium.com/@taposhdr/forecasting-methods-part-i-9440e27466ab)
3. [Time series forecasting: from naive to ARIMA and beyond](https://towardsdatascience.com/time-series-forecasting-from-naive-to-arima-and-beyond-ef133c485f94)
