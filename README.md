# Kaggle Digit Recognizer

Solution to [Kaggle Digit Recognizer challenge](https://www.kaggle.com/c/digit-recognizer).

## Usage
Run train.py

then test.py

## Solution
The training set contains images, each having a resolution of (28x28) 784 pixels. Therefore every image is a square of 28 rows and columns

num_classes is the number of classes the images should be classified into

as the train.csv is actually a set of 42000 images from 0-9, these 42000 images need to be classified into 10 distinct classes

the out_y specifies the classes, and out_x contains the data to be classified

### Sequential
I chose the sequential model as there is exactly one input tensor and one output tensor.

I chose the golden ratio (80:20) to create the training source (33,600 samples) and a validation source (8,400 samples).

## Extra

The trained model is picklable and is stored in `output` directory.
