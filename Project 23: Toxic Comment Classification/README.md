## PROJECT 23: Toxic Comment Classification

> ### TASK: Identify and classify toxic online comments

### Project goals and objectives

#### Project goal

- Studying **transfer learning for NLP**

#### Project objectives

1. Explore and prepare data 
2. Use Tensorflow HUB and Universal Sentence Encoder 

### Dataset

[Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

**DATASET INFORMATION:**


##### Context
You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.


##### Content
- `train.csv` - the training set, contains comments with their binary labels
- `test.csv` - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
- `sample_submission.csv` - a sample submission file in the correct format
- `test_labels.csv` - labels for the test data; value of -1 indicates it was not used for scoring

### Results

1. [x] [**Universal Sentence Encoder**]()


### References

1. [Universal Sentence Encoder for English](https://aclanthology.org/D18-2029.pdf)
2. [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder/4)
