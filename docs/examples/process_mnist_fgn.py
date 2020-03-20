from time import time
import numpy as np
from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
import tensorflow as tf

"""
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

digits = datasets.load_digits(n_class=6)
X = inputs_test_images
y = outputs_test_labels
n_samples, n_features = X.shape
n_neighbors = 30

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    mean = []
    for i in range(0, 10):
        mean.append(X[np.where(outputs_test_labels == i)].mean(axis=0))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    for i in range(0, 10):
        plt.annotate(''+str(i), xy=(mean[i][0], mean[i][1]), xytext=(mean[i][0], mean[i][1]), fontsize = 20)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, verbose= 1, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(np.array(inputs_test_images))


plot_embedding(X_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))

plt.show()

"""

def measure_groups(clusters, y):

    for c_id, c in enumerate(clusters):
        print('Cluster # {} Elements {}'.format(c_id, len(c)))
        y_s = []
        for i in c:
            y_s.append(y[i])
        print(np.unique(y_s, return_counts=True))


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
# metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())

data = x_train.copy()[:10000]

# data = np.ascontiguousarray(data.reshape(data.shape[0], 28 * 28))
data = np.ascontiguousarray(data / 255.)
feature_shape = (28, 28)

from pdb import set_trace

# set_trace()
thresholds = [784 * 0.01 * 0.025,
              784 * 0.01 * 0.020,
              784 * 0.01 * 0.015,
              784 * 0.01 * 0.010]

thresholds = [784 * 0.01 * 0.010]

interactive = True

fgn = FastGroupingNode(thresholds, metric)

fgn_next = fgn.cluster_online(feature_shape)

print('Async FGN version')

t1 = time()
for i in range(len(data)):
    fgn_online_tree, path = fgn_next(data[i], i)
t2 = time()
print('Duration %.3f s' % (t2 - t1,))

fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

for i in range(len(thresholds)):
    print('Level', i + 1)
    clusters = fgn_online_all_levels.get_clusters(i + 1)
    if interactive:
        print('# clusters', len(clusters))
        print('clusters', clusters)
    print(len(clusters))



