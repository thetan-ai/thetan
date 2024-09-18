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

from thetan.berserker import berserker2

centroids, labels = berserker2(x, theta_value=8.5, metric='euclidean', all_out=False, verbose=True)

t = 1

import matplotlib.pyplot as plt

for i in range(18):
    plt.figure()
    plt.title('Label ' + str(i))
    plt.imshow(centroids[i].reshape(28, 28))
    plt.show()
