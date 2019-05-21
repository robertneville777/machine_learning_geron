# function to download popular datasets, specifically MNIST
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original') # download MNIST dataset
X,y = mnist["data"],mnist["target"] # separate data

# MNIST dataset is 70000 images of handwritten digits (0-9), that are in 28x28
# images. Each image is 28x28=784 pixels, each pixel assigned a gray scale
# from 0 - 255. So each row is one image, and each value in that row is a
# pixel's gray scale value. There are 70000 rows/images and 784 columns/gray-
# scale-values per row. 

some_digit = X[36000] # Let's arbitrarily look at the 36000th image in the 
                      # dataset.
some_digit_image = some_digit.reshape(28,28) # reshape the grayscale values
                                             # in the appropriate image dimen-
                                             # sions.

import matplotlib # import plotting functions
import matplotlib.pyplot as plt

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show() # plot image

y[36000]  # image's digit that is portraying

# Split dataset in testing and training datasets. MNIST already comes separated
# into testing and training datasets: the first 60k images are training, and
# the last 10k are testing.

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Shuffle training sets. Good idea since some learning algorithms are sensitive
# to the order of training sets. Not always a good idea, for example, on time
# series data.

import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Creating a "5-detector". Create the target vectors for the classification
# task:
y_train_5 = (y_train == 5) # true for all 5s, False for all other digits.
y_test_5  = (y_test  == 5)

# Pick a classifier to train. Will use Stochastic Gradient Descent.

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Use model to detect images of 5s:
sgd_clf.predict([some_digit]) # output is true, the classifier properly pred-
                              # icts that the image is 5.