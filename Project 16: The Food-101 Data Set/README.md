## PROJECT 16: Food 101 


> ### TASKS: Classification

### Project goals and objectives

#### Project goal

- Studying **feature extraction** transfer learning 
- Studying Tensorflow Datasets 

#### Project objectives

1. Using TensorFlow Datasets to download and explore data
2. Building a feature extraction model
3. Building a fine-tuning model
4. Viewing training results on TensorBoard

### Dataset

[The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

**DATASET INFORMATION:**

We introduce a challenging data set of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.


### Results

1. [x] [**Feature extraction model: EfficientNet**](https://github.com/rttrif/TrifonovRS.Deep_Learning_Portfolio.github.io/blob/main/Project%2016:%20The%20Food-101%20Data%20Set/EfficientNet_Feature_extraction.py)
2. [x] [**TensorBoard for feature extraction model**](https://tensorboard.dev/experiment/B0P2g4tkTkSb9j6U3gsEmQ/#scalars)
3. [x] [**Fine-tuning model: RegNetY 400MF**](https://github.com/rttrif/TrifonovRS.Deep_Learning_Portfolio.github.io/blob/main/Project%2016:%20The%20Food-101%20Data%20Set/RegNetY_Fine_tuning.py)
4. [x] [**TensorBoard for fine-tuning model**]()


### References

1. [Food-101 – Mining Discriminative Components with Random Forests](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
2. [Instructions for transfer learning with pre-trained CNNs](https://medium.com/@mikhaillenko/instructions-for-transfer-learning-with-pre-trained-cnns-203ddaefc01)
3. [Transfer Learning in Tensorflow (VGG19 on CIFAR-10): Part 1](https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4)
4. [Transfer Learning in Tensorflow (VGG19 on CIFAR-10): Part 2](https://towardsdatascience.com/transfer-learning-in-tensorflow-5d2b6ad495cb)
5. [RegNetY](https://github.com/AdityaKane2001/regnety)
