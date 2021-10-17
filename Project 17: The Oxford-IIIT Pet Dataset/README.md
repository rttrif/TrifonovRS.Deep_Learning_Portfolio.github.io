## PROJECT 17: The Oxford-IIIT Pet Dataset


> ### TASKS: Semantic Segmentation

### Project goals and objectives

#### Project goal

- Studying **transfer learning**
- Studying Tensorflow Datasets 

#### Project objectives

1. Using TensorFlow Datasets to download and explore data
2. Building a transfer model
3. Viewing training results on TensorBoard

### Dataset

[The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

**DATASET INFORMATION:**

We have created a 37 category pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.


### Results

1. [x] [**Fine-tuning model: U-net**](https://github.com/rttrif/TrifonovRS.Deep_Learning_Portfolio.github.io/blob/main/Project%2017:%20The%20Oxford-IIIT%20Pet%20Dataset/U_net_Transfer_learning.py)
2. [x] [**TensorBoard for fine-tuning model**](https://tensorboard.dev/experiment/M9poPTZ8SO6QOHZyLZsxlQ/#scalars)



### References

1. [Cats and Dogs](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)
2. [Instructions for transfer learning with pre-trained CNNs](https://medium.com/@mikhaillenko/instructions-for-transfer-learning-with-pre-trained-cnns-203ddaefc01)
3. [Transfer Learning in Tensorflow (VGG19 on CIFAR-10): Part 1](https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4)
4. [Transfer Learning in Tensorflow (VGG19 on CIFAR-10): Part 2](https://towardsdatascience.com/transfer-learning-in-tensorflow-5d2b6ad495cb)
