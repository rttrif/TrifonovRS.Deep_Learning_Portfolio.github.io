## PROJECT 15: Car segmentation


> ### TASK: Semantic Segmentation

### Project goals and objectives

#### Project goal

Studying architecture: **Mask R-CNN**


#### Project objectives

1. Exploratory Data Analysis
2. Training Mask R-CNN

### Dataset

[Car segmentation](https://www.kaggle.com/intelecai/car-segmentation)

**DATASET INFORMATION:**

This is a sample semantic image segmentation dataset. It contains images of cars and their segmentation masks. Most of the images were taken from side of the car. Image and its corresponding mask have the same name. For example, 003.png in the "masks" folder corresponds to the 003.png file in the "images" folder. Each pixel in a mask shows the class of the corresponding pixel in the corresponding image. For example, if value of pixel (3, 7) is 1, it means pixel (3,7) in the corresponding image belongs to class 1. We have following 5 classes in this dataset:

0 - background
1 - car
2 - wheel
3 - light
4 - windows
Half of the images were collected from the Internet (especially from unsplash.com) and another half were taken in the streets.


### Results

1. [ ] [**Mask R-CNN**]()


### References

1. [Building a Mask R-CNN from scratch in TensorFlow and Keras](https://towardsdatascience.com/building-a-mask-r-cnn-from-scratch-in-tensorflow-and-keras-c49c72acc272)
2. [Mask R-CNN (Original papar)](https://arxiv.org/pdf/1703.06870.pdf)
3. [Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
4. [Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN)
