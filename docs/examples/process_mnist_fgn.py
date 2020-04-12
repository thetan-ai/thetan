from time import time
import numpy as np
from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def report_confusion_matrix(y_true, y_pred):
    cmat = confusion_matrix(y_true, y_pred,
                            labels=None,
                            sample_weight=None,
                            normalize=None)
    return cmat


def report_accurracy(y_true, y_pred):

    accs = accuracy_score(y_true, y_pred,
                          normalize=True,
                          sample_weight=None)
    print(accs)


def report_groups(clusters, y):

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

thresholds = [0.1]

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
        # print('clusters', clusters)
    print(len(clusters))
    report_groups(clusters, y_train)

