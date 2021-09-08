"""
PROJECT GOALS AND OBJECTIVES

PROJECT GOAL
Development of skills for building and improving neural network models for binary classification
using the sequential and functional Tensorflow API

STAGE OBJECTIVES
1. Data preparation

DATASET: Heart Failure Prediction

ATTRIBUTE INFORMATION:

1. age: Age of the patient

2. anaemia: If the patient had the haemoglobin below the normal range

3. creatinine_phosphokinase: The level of the creatine phosphokinase in the blood in mcg/L

4. diabetes: If the patient was diabetic

5. ejection_fraction: Ejection fraction is a measurement
of how much blood the left ventricle pumps out with each contraction

6. high_blood_pressure: If the patient had hypertension

7. platelets: Platelet count of blood in kiloplatelets/mL

8. serum_creatinine: The level of serum creatinine in the blood in mg/dL

9. serum_sodium: The level of serum sodium in the blood in mEq/L

10. sex: The sex of the patient

11. smoking: If the patient smokes actively or ever did in past

12. time: It is the time of the patient's follow-up visit for the disease in months

13. DEATH_EVENT: If the patient deceased during the follow-up period
"""
# %%
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
# READ DATA

data_path = "/Users/rttrif/Data_Science_Projects/Tensorflow_Certification/" \
            "Project_2_ Heart_Failure_Prediction/data/heart_failure_clinical_records_dataset.csv"

data = pd.read_csv(data_path)

data_head = data.head().T

data_tail = data.tail().T

# %%
# DATA EXPLORATION

descriptive_statistics = data.describe().T

info = data.info()

missing_values = data.isna().sum()

# %%
# VISUALIZATION

# Evaluation of the target variable: DEATH_EVENT
sns.countplot(x=data["DEATH_EVENT"])
plt.show()

# Correlation matrix of all the features
sns.set_theme(style="white")
corr_matrix = data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9, )
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, center=0, annot=True, fmt='.2f',
            square=True, linewidths=2, cbar_kws={"shrink": .5})
plt.show()

"""
RESULTS OF THE ANALYSIS CORRELATION MATRIX

1. Time of the patient's follow-up visit for the disease is crucial in as initial diagnosis with cardiovascular issue 
and treatment reduces the chances of any fatality. It holds and inverse relation.

2. Ejection fraction is the second most important feature. 
It is quite expected as it is basically the efficiency of the heart.

3.Age of the patient is the third most correlated feature. Clearly as heart's functioning declines with ageing
"""

# Evaluating age distribution
plt.figure(figsize=(20, 12))
Days_of_week = sns.countplot(x=data['age'], data=data, hue="DEATH_EVENT")
Days_of_week.set_title("Distribution Of Age", color="#774571")
plt.show()

# Boxen and swarm plot of some non binary features.
feature = ["age", "creatinine_phosphokinase", "ejection_fraction",
           "platelets", "serum_creatinine", "serum_sodium", "time"]
for i in feature:
    plt.figure(figsize=(10, 10))
    sns.swarmplot(x=data["DEATH_EVENT"], y=data[i], color="black", alpha=0.5)
    sns.boxenplot(x=data["DEATH_EVENT"], y=data[i])
    plt.show()

"""
The data set has a significant number of outliers, which ultimately can lead to retraining of the model.

However, given the specifics of medical data, it is better to leave outliers 
because they can have a significant statistical value.
"""

# A kernel density estimate of time and age
sns.kdeplot(x=data["time"], y=data["age"], hue=data["DEATH_EVENT"])
plt.show()

# %%
# DATA PREPARATION

# Splitting into target variable and feathers
X = data.drop(["DEATH_EVENT"], axis=1)
y = data["DEATH_EVENT"]

# %%
# NORMALIZE DATA

# scaler
scaler = StandardScaler()

# Normalize
col_names = list(X.columns)
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=col_names)

descriptive_statistics_X_scaled = X_scaled.describe().T

# Visualisation of scaled data
plt.figure(figsize=(20,10))
sns.boxenplot(data=X_scaled)
plt.xticks(rotation=90)
plt.show()

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

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