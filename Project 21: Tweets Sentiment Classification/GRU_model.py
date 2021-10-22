"""
PROJECT 21: Tweets Sentiment Classification
TASK: Natural Language Processing to identify the sentiments
PROJECT GOALS AND OBJECTIVES
PROJECT GOAL
- Studying Recurrent neural network for NPL
- Studying architecture LSTM
- Studying architecture GRU
- Studying architecture Bidirectional-LSTM
PROJECT OBJECTIVES
1. Explore and prepare data
2. Training simple RNN model
3. Training LSTM
4. Training GRU
5. Training Bidirectional-LSTM
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
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dropout, Dense, Input, Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model

# %%
# Path to data
train_data_path = "data/train_2kmZucJ.csv"
test_data_path = "data/test_oJQbWVk.csv"

# Create dataframe
train_df = pd.read_csv(train_data_path, index_col=0)
test_df = pd.read_csv(test_data_path)

train_df.head()
test_df.head()
# %%
train_df.info()
train_df['label'].value_counts()
# %%
# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.head()
# %%
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")
# %%
# Random training examples
random_index = random.randint(0, len(train_df) - 5)
for row in train_df_shuffled[["label", "tweet"]][random_index:random_index + 5].itertuples():
    _, label, tweet = row
    print(f"Target: {label}", "(negative)" if label > 0 else "(not negative)")
    print(f"Text:\n{tweet}\n")
    print("---\n")
# %%
# Split data
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["tweet"].to_numpy(),
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


def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                                                 labels=np.unique(y_pred))
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


# %%
# MULTINOMIAL NAIVE BAYES CLASSIFIER LIKE BASELINE MODEL
model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model_0.fit(train_sentences.astype('str'), train_labels)

# %%
# EVALUATION RESULTS BASELINE MODEL
baseline_score = model_0.score(val_sentences, val_labels)
print(f"Model accuracy: {baseline_score * 100:.2f}%")

# %%
# Predictions
baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]

# %%
baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
baseline_results

# %%
# SIMPLE RNN MODEL
max_vocab_length = 10000
max_length = 50

# %%
# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels))

train_dataset

# %%
# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

train_dataset

# TextVectorization layer
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

text_vectorizer.adapt(train_sentences.astype('str'))

# Embedding Layer
embedding = Embedding(input_dim=max_vocab_length,
                      output_dim=128,
                      embeddings_initializer="uniform",
                      input_length=max_length,
                      name="embedding_1")

# GRU Model
inputs = Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)

x = GRU(512, return_sequences=True)(x)
x = Dropout(0.5)(x)

x = GRU(256, return_sequences=True)(x)
x = Dropout(0.5)(x)

x = GRU(128, return_sequences=True)(x)
x = Dropout(0.3)(x)

x = GRU(64, return_sequences=True)(x)
x = Dropout(0.25)(x)

x = GRU(32, return_sequences=True)(x)
x = Dropout(0.25)(x)

x = GRU(16)(x)

outputs = Dense(1, activation="sigmoid")(x)

model_GRU = Model(inputs, outputs, name="GRU_model")

# Compile model
model_GRU.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

model_GRU.summary()
tf.keras.utils.plot_model(model_GRU, to_file='GRU_model.png')

# %%
# Train model
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                     restore_best_weights=True)

model_GRU_history = model_GRU.fit(train_sentences,
                                  train_labels,
                                  epochs=200,
                                  validation_data=(val_sentences, val_labels),
                                  callbacks=[early_stopping_cb])
# %%
# EVALUATION RESULT
# Learning curves
learning_curves(model_GRU_history)

# Evaluation model
evaluation_model(model_GRU_history)

# Check the results
model_GRU.evaluate(val_sentences, val_labels)

# Predictions
model_GRU_pred_probs = model_GRU.predict(val_sentences)
model_GRU_preds = tf.squeeze(tf.round(model_GRU_pred_probs))
model_GRU_preds[:20]

model_GRU_results = calculate_results(y_true=val_labels,
                                      y_pred=model_GRU_preds)
model_GRU_results

# Compare model results
model_results = pd.DataFrame({"baseline": baseline_results,
                              "GRU_model": model_GRU_results})
