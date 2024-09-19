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

# x = np.concatenate((x_train, x_test))
# y = np.concatenate((y_train, y_test))  ## labels (used for verification)
# x = x.reshape((x.shape[0], -1))
# x = np.divide(x, 255.)
# n_clusters = len(np.unique(y))
# print(n_clusters)


# from sklearn import datasets, manifold

# digits = datasets.load_digits()
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape

from thetan.berserker import berserker2
from thetan.theta import theta_g1, assign, centroids_from_labels

x_train_cp = np.ascontiguousarray(x_train.reshape(60000, 28*28), dtype='f8')
x_test_cp = np.ascontiguousarray(x_test.reshape(10000, 28*28), dtype='f8')

# centroids_train_avgs = centroids_from_labels(x_train_cp, x_test.astype('i8'))



# centroids, labels = berserker2(x_train_cp, theta_value=1900., metric='euclidean', all_out=False, verbose=True)

centroids_train = []
corresponding_labels_train = []
# L2
# theta_value = 1800.
# theta_value = 1700.
# metric = 'euclidean'

# Dot Product
# theta_value = 100.
# metric = 'dot_prod'

# L1
theta_value = 18000.
metric = 'l1_norm'


for y_t in range(10):
    x_tmp = np.ascontiguousarray(x_train_cp[y_train == y_t].copy())
    centroids, labels = berserker2(x_tmp, theta_value=theta_value, metric=metric, all_out=False, verbose=False)
    centroids_train.append(centroids)
    corresponding_labels_train.extend(centroids.shape[0] * [y_t])
    print(y_t, centroids.shape)


centroids_train = np.ascontiguousarray(np.concatenate(centroids_train))

# dot product needs work - it creates NaNs for ber
# centroids, labels = berserker2(x, theta_value=80, metric='dot_prod', all_out=False, verbose=True)

# TS does not create NaN but not helping much
# centroids, labels = theta_g1(x, theta=70, metric='dot_prod', all_out=False, verbose=False)
print('Number of centroids found ', centroids_train.shape[0])

t = 1

# plot_centroid_images(centroids)


t = 2


labels_x_train = assign(x_train_cp, centroids_train, metric=metric)

y_train_pred = []
for i, l in enumerate(labels_x_train):
    y_train_pred.append(corresponding_labels_train[l])
y_train_pred = np.ascontiguousarray(y_train_pred, dtype='i8')

labels_x_test = assign(x_test_cp, centroids_train, metric=metric)

y_test_pred = []
for i, l in enumerate(labels_x_test):
    y_test_pred.append(corresponding_labels_train[l])
y_test_pred = np.ascontiguousarray(y_test_pred, dtype='i8')


ac_train = accuracy_score(y_train, y_train_pred)

ac_test = accuracy_score(y_test, y_test_pred)

print('Reporting scores ...')
print('AC Train ', ac_train, ' AC Test ', ac_test)

t = 3