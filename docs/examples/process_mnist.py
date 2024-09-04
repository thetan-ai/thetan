import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))  ## labels (used for verification)
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
n_clusters = len(np.unique(y))
print(n_clusters)
x.shape

# from sklearn import datasets, manifold

# digits = datasets.load_digits()
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape


