
# Machine Learning Engineer Nanodegree
## Deep Learning
## Project: Recognizing digits from MNIST and SVHN dataset


**Aim**:

Here, we want to train a algorithm on a synthetic dataset created by concatenating digits from MNIST dataset, and use that trained algorithm to predict digits in the SVHN dataset.

**Approach taken**:

I used a convolutional neural network for this problem, since it has been touted as one of the best methods for image classification.
First, MNIST dataset consists only of single digits. A dataset containing five digits had to be created. For this first it was required to convert the dataset from idx to numpy format. This was done with the help was of a package called idx2numpy (https://pypi.python.org/pypi/idx2numpy), information regarding which was obtained from this thread in forums:https://discussions.udacity.com/t/questions-about-step-1/204239.
For feeding the data into the CNN, my theory was that since each element in dataset consists of 5 28 x 28 images stacked together side-to-side, at a time, widthwise only 28 pixels of the image will be fed to the model,(example: from 0 to 28). Since total width is 140, 5 classifiers will have to to be used and their cumulative loss function will be considered.

For the SVHN dataset, a CNN model model was used as in the synthetic MNIST dataset, but with the addition of a dropout layer with a probability of 0.9. Similar to MNIST problem, 5 separate classifiers were used. However, number of labels possible in a particular position was 11, as it could be possible that there was no labels present. Optimizer was changed from GradientDescent to AdamOptimizer with an exponentially decaying rate.
The model does not perform as well on a realistic dataset as compared to the synthetic dataset. Batch size has been increased to 64 and an additional dropout layer has been added. Still, the accuracy hovers around 62 to 65% for training set for different settings of learning rates. I think this can be improved by increasing the training examples and/or training steps, but even these existing claculations take up a lot of time on my system.
