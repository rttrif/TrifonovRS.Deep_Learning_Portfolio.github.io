"""
PROJECT 20: Sms spam collection
TASK: Natural Language Processing to predict a SMS is spam or not spam
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
- Studying Feed-forward neural network for NPL
- Studying tokenization, text vectorisation and embedding
PROJECT OBJECTIVES
1. Exploratory Data Analysis
2. Training several dense model
3. Predict a SMS is spam or not spam
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

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# %%
# LOAD AND EXPLORE DATASET

# Path to data
train_data_path = "data/train.csv"
test_data_path = "data/test.csv"

# Create dataframe
train_df = pd.read_csv(train_data_path, index_col=0)
test_df = pd.read_csv(test_data_path, index_col=0)
train_df.head()
# %%
train_df.columns = ["label","message"]
train_df.info()
train_df['label'].value_counts()
# %%
code_spam_ham = {"ham": 0,
                 "spam": 1}

train_df['label'] = train_df['label'].map(code_spam_ham)
train_df.head()
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
random_index = random.randint(0, len(train_df) - 5)
for row in train_df_shuffled[["label", "message"]][random_index:random_index + 5].itertuples():
    _, label, message = row
    print(f"Target: {label}", "(spam)" if label > 0 else "(not spam)")
    print(f"Text:\n{message}\n")
    print("---\n")
# %%
# Split data
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["message"].to_numpy(),
                                                                            train_df_shuffled["label"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)

# Check the lengths
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)

# %%
# The first 10 training sentences and their labels
train_sentences[:10]
train_labels[:10]

# %%
# MULTINOMIAL NAIVE BAYES CLASSIFIER
model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model_0.fit(train_sentences.astype('str'), train_labels)


# %%
# EVALUATION RESULTS
baseline_score = model_0.score(val_sentences, val_labels)
print(f"Model accuracy: {baseline_score*100:.2f}%")

# %%
# Predictions
baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]

# %%
def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", labels=np.unique(y_pred))
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
baseline_results

# %%
# EVALUATION AND VISUALIZATION OF MODEL PARAMETERS

def learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(20, 8))
    plt.grid(True)
    plt.title('Learning curves')
    plt.gca().set_ylim(0, 1)
    plt.show()


def evaluation_model(history):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(20, 8))
    axL.plot(history.history['loss'], label="Training loss")
    axL.plot(history.history['val_loss'], label="Validation loss")
    axL.set_title('Training and Validation loss')
    axL.set_xlabel('Epochs')
    axL.set_ylabel('Loss')
    axL.legend(loc='upper right')

    axR.plot(history.history['accuracy'], label="Training accuracy")
    axR.plot(history.history['val_accuracy'], label="Validation accuracy")
    axR.set_title('Training and Validation accuracy')
    axR.set_xlabel('Epoch')
    axR.set_ylabel('Accuracy')
    axR.legend(loc='upper right')

    plt.show()
# %%
# DENSE MODEL №1
max_vocab_length = 10000
max_length = 50
# TextVectorization layer
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

text_vectorizer.adapt(train_sentences.astype('str'))

# Embedding Layer
embedding = layers.Embedding(input_dim=max_vocab_length,
                             output_dim=128,
                             embeddings_initializer="uniform",
                             input_length=max_length,
                             name="embedding_1")

# Model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")


model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_1.summary()
tf.keras.utils.plot_model(model_1, to_file='model_1_dense.png')
# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3,
                                                     restore_best_weights=True)
model_1_history = model_1.fit(train_sentences.astype('str'),
                              train_labels,
                              epochs=200,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(model_1_history)

# Evaluation model
evaluation_model(model_1_history)

# Check the results
model_1.evaluate(val_sentences, val_labels)

# Predictions
model_1_pred_probs = model_1.predict(val_sentences)
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_preds[:20]

model_1_results = calculate_results(y_true=val_labels,
                                    y_pred=model_1_preds)
model_1_results

# %%
# DENSE MODEL №2
# Model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs, name="model_2_dense")


model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2.summary()
tf.keras.utils.plot_model(model_2, to_file='model_2_dense.png')
# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3,
                                                     restore_best_weights=True)
model_2_history = model_2.fit(train_sentences.astype('str'),
                              train_labels,
                              epochs=200,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(model_2_history)

# Evaluation model
evaluation_model(model_2_history)

# Check the results
model_2.evaluate(val_sentences, val_labels)

# Predictions
model_2_pred_probs = model_2.predict(val_sentences)
model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:20]

model_2_results = calculate_results(y_true=val_labels,
                                    y_pred=model_2_preds)
model_2_results

# %%
# DENSE MODEL №3
# Model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs, name="model_3_dense")


model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.summary()
tf.keras.utils.plot_model(model_3, to_file='model_3_dense.png')
# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3,
                                                     restore_best_weights=True)
model_3_history = model_3.fit(train_sentences.astype('str'),
                              train_labels,
                              epochs=200,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(model_3_history)

# Evaluation model
evaluation_model(model_3_history)

# Check the results
model_3.evaluate(val_sentences, val_labels)

# Predictions
model_3_pred_probs = model_3.predict(val_sentences)
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:20]

model_3_results = calculate_results(y_true=val_labels,
                                    y_pred=model_3_preds)
model_3_results