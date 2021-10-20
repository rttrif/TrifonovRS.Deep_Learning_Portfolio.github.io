"""
PROJECT 19: Natural Language Processing with Disaster Tweets
TASK: Natural Language Processing
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
- Studying Multinomial Naive Bayes algorithm.
- Studying TF-IDF (term frequency-inverse document frequency)
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training multinomial Naive Bayes
3. Predict which Tweets are about real disasters and which ones are not
"""
# %%
# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# %%
# LOAD AND EXPLORE DATASET

# Path to data
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"

# Create dataframe
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
train_df.head()
# %%
train_df.info()
train_df.target.value_counts()
# %%
# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.head()
# %%
test_df.head()
test_df.info()
# %%
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")
# %%
# Random training examples
random_index = random.randint(0, len(train_df) - 5)  # create random indexes not higher than the total number of samples
for row in train_df_shuffled[["text", "target"]][random_index:random_index + 5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
    print(f"Text:\n{text}\n")
    print("---\n")

# %%
# Split data
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)

# Check the lengths
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)

# %%
# The first 10 training sentences and their labels
train_sentences[:10], train_labels[:10]

# %%
# MULTINOMIAL NAIVE BAYES CLASSIFIER
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(train_sentences, train_labels)


# %%
# EVALUATION RESULTS
baseline_score = model.score(val_sentences, val_labels)
print(f"Model accuracy: {baseline_score*100:.2f}%")

# %%
# Predictions
baseline_preds = model.predict(val_sentences)
baseline_preds[:20]

# %%
def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
print(baseline_results)