import numpy as np
from keras.datasets import mnist

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def report_accuracy(y_true, y_pred):

    accs = accuracy_score(y_true, y_pred,
                          normalize=True,
                          sample_weight=None)
    return accs

def plot_centroid_images(centroids):
    num_centroids = centroids.shape[0]
    f, axarr = plt.subplots(1, num_centroids)

    for i in range(num_centroids):
        axarr[i].imshow(centroids[i].reshape(28, 28), cmap='gray')

    plt.show()


def plot_centroid_images_obo(centroids):
    for i in range(centroids.shape[0]):
        plt.figure()
        plt.title('Label ' + str(i))
        plt.imshow(centroids[i].reshape(28, 28))
        plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))  ## labels (used for verification)
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
n_clusters = len(np.unique(y))
print(n_clusters)


# from sklearn import datasets, manifold

# digits = datasets.load_digits()
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape

from thetan.berserker import berserker2
from thetan.theta import theta_g1, assign, centroids_from_labels

x_train_cp = np.ascontiguousarray(x_train.reshape(60000, 28*28), dtype='f8')

centroids_train = centroids_from_labels(x_train_cp, y_train.astype('i8'))

centroids, labels = berserker2(x_train_cp, theta_value=2000., metric='euclidean', all_out=False, verbose=True)



# dot product needs work - it creates NaNs for ber
# centroids, labels = berserker2(x, theta_value=80, metric='dot_prod', all_out=False, verbose=True)

# TS does not create NaN but not helping much
# centroids, labels = theta_g1(x, theta=70, metric='dot_prod', all_out=False, verbose=False)

print('Number of centroids found ', centroids.shape[0])

t = 1

plot_centroid_images(centroids)


t = 2

 
labels_x = assign(x, centroids, metric='euclidean')



labels_y = assign(y, centroids, metric='euclidean')

accuracy_score(y_train, labels_x)

accuracy_score(y_test, labels_y)