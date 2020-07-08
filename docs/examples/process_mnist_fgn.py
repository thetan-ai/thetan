from pdb import set_trace
from time import time
import numpy as np
from skimage import filters
from robotikos.clustering import FastGroupingNode, ClusterMapCentroid, ClusterCentroid
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from scipy.linalg import qr, svd
from robotikos.fgm import fgn_ma, fgn_shuffles


logger = logging.getLogger(__name__)


def report_confusion_matrix(y_true, y_pred):
    cmat = confusion_matrix(y_true, y_pred,
                            labels=None,
                            sample_weight=None,
                            normalize=None)
    return cmat


def report_accuracy(y_true, y_pred):

    accs = accuracy_score(y_true, y_pred,
                          normalize=True,
                          sample_weight=None)
    return accs


def report_exactness(y_true, y_pred):
    exactness = np.sum(np.abs(y_true - y_pred) == 0) / len(y_true)
    return exactness


def report_unique(clusters, y_true):
    """ use with caution only repeating most predicted value"""

    y_pred = []
    for c_id, c in enumerate(clusters):
        print('Cluster # {} Elements {}'.format(c_id, len(c)))
        y_s = []
        for i in c:
            y_s.append(y_true[i])
        elements, counts = np.unique(y_s, return_counts=True)
        # print(elements)
        # print(counts)
        pred_value = elements[np.argmax(counts)]
        # print('classified as ' + str(pred_value))
        pred_value_ext = [pred_value] * len(y_s)
        y_pred.extend(pred_value_ext)
        # set_trace()
        y_sa = np.asarray(y_s)
        print(np.sum(y_sa == pred_value) / len(y_sa))
    return np.array(y_pred)


def separate_train_data(x_train, y_train):
    # TODO be careful with this one it goes only up to 10
    """ Separates training data to each category"""

    sep_data = {}
    for i in range(10):
        sep_data[i] = x_train[y_train == i]

    return sep_data


dataset_type = 'mnist'

if dataset_type == 'mnist':
    print('Loading Hand Written Digits')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset_type == 'fashion':
    print('Loading Fashion MNIST dataset')
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
else:
    raise ValueError('Uknown dataset type')


metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
# metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())

train_sz = 60000
test_sz = 10000
use_sobel = False
print('Training size', train_sz)
print('Test size', test_sz)

# thresholds = [0.1]
thresholds = [0.5, 0.2, 0.08, 0.05]
# thresholds = [0.05]
# Accuracy training set 0.7715833333333333
# Accuracy test set (min_dist) 0.7503
print('Thresholds', thresholds)

# TODO add data shuffling

x_train_part = x_train.copy()[:train_sz]
x_train_part = np.ascontiguousarray(x_train_part / 255.)
if use_sobel:
    for i in range(train_sz):
        x_train_part[i] = filters.sobel(x_train_part[i])

x_test_part = x_test.copy()[:test_sz]
x_test_part = np.ascontiguousarray(x_test_part / 255.)
if use_sobel:
    for i in range(test_sz):
        x_test_part[i] = filters.sobel(x_test_part[i])

y_train_part = y_train.copy()[:train_sz]
y_test_part = y_test.copy()[:test_sz]

feature_shape = x_train_part[0].shape


def count_cluster_spread():
    """ This will not be used as is """

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
        # y_pred = report_unique(clusters, y_train)
        # report_accuracy(y_train, y_pred)


def fsn_training_no_fgn(x_train, y_train, metric):
    sep_data = separate_train_data(x_train, y_train)

    centroids = {}
    for i in sep_data:
        centroids[i] = np.mean(sep_data[i], axis=0)

    y_pred = y_pred_no_fgn(x_train, centroids, metric)
    return centroids, y_pred


def fsn_test_no_fgn(x_test, centroids, metric):
    return y_pred_no_fgn(x_test, centroids, metric)


def y_pred_no_fgn(x_features, centroids, metric):
    y_pred = []
    for x in x_features:
        min_dists = []
        for i in centroids:
            c = centroids[i]
            d = metric.dist(x.astype('f4'), c.astype('f4'))
            min_dists.append(d)
        y_pred.append(np.argmin(min_dists))
    return y_pred


def fsn_training_with_fgn(x_train, y_train, thresholds, metric):

    sep_data = separate_train_data(x_train, y_train)

    print('Async FGN version')
    print('Thresholds', thresholds)

    centroids = {}
    for s in sep_data:
        t1 = time()

        fgn = FastGroupingNode(thresholds, metric)
        fgn_next = fgn.cluster_online(feature_shape)

        for i in range(sep_data[s].shape[0]):
            fgn_online_tree, _path = fgn_next(sep_data[s][i], i)

        t2 = time()
        print('Duration %.3f s' % (t2 - t1,))

        fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

        for t in range(len(thresholds)):
            print('Level', t + 1)
            clusters = fgn_online_all_levels.get_clusters(t + 1)

        centroids[s] = clusters.centroids

    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])
    print('Total number of centroids', total_no_centroids)

    y_pred = y_pred_with_fgn(x_train, centroids, metric)

    return centroids, y_pred


def fsn_training_with_fgn_ma(x_train, y_train, thresholds, metric):

    sep_data = separate_train_data(x_train, y_train)

    print('Async FGN version')
    print('Thresholds', thresholds)

    centroids = {}
    for s in sep_data:
        # t1 = time()

        # fgn = FastGroupingNode(thresholds, metric)
        # fgn_next = fgn.cluster_online(feature_shape)

        # for i in range(sep_data[s].shape[0]):
        #     fgn_online_tree, _path = fgn_next(sep_data[s][i], i)

        # t2 = time()
        # print('Duration %.3f s' % (t2 - t1,))

        # fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

        # for t in range(len(thresholds)):
        #     print('Level', t + 1)
        #     clusters = fgn_online_all_levels.get_clusters(t + 1)

        # centroids[s] = clusters.centroids

        all_levels = fgn_ma(sep_data[s], thresholds, metric)
        allk = sorted(all_levels.keys())
        centroids[s] = all_levels[allk[-1]].centroids

    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])
    print('Total number of centroids', total_no_centroids)

    y_pred = y_pred_with_fgn(x_train, centroids, metric)

    return centroids, y_pred


def fsn_training_with_fgn_shuffles(x_train, y_train, thresholds, metric):

    sep_data = separate_train_data(x_train, y_train)

    print('Async FGN version')
    print('Thresholds', thresholds)

    centroids = {}
    for s in sep_data:

        collect2 = fgn_shuffles(sep_data[s], thresholds, metric=metric,
                                num_shuffles=2, mode='fgn_ma')
        centroids[s] = collect2[len(thresholds)]['centroids']

    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])
    print('Total number of centroids', total_no_centroids)

    y_pred = y_pred_with_fgn_shuffles(x_train, centroids, metric)

    return centroids, y_pred


def fsn_test_with_fgn(x_test, centroids, metric):
    return y_pred_with_fgn(x_test, centroids, metric)


def y_pred_with_fgn(x_features, centroids, metric):
    y_pred = []
    cnt = 0
    for x in x_features:
        if cnt % 5000 == 0:
            print(cnt)
        min_dists = []
        # there are multiple centroids for each underlying training class
        for key in centroids:
            min_dists_category = []
            # look at all available centroids
            for c in centroids[key]:
                d = metric.dist(x.astype('f4'), c.astype('f4'))
                # d = metric.dist(x.astype('f4'), centroids[key][c].astype('f4'))
                min_dists_category.append(d)
            min_dists.append(np.min(min_dists_category))
        y_pred.append(np.argmin(min_dists))
        cnt += 1
    return y_pred


def y_pred_with_fgn_shuffles(x_features, centroids, metric):
    y_pred = []
    cnt = 0
    for x in x_features:
        if cnt % 5000 == 0:
            print(cnt)
        min_dists = []
        # there are multiple centroids for each underlying training class
        for key in centroids:
            min_dists_category = []
            # look at all available centroids
            for c in centroids[key]:
                # d = metric.dist(x.astype('f4'), c.astype('f4'))
                d = metric.dist(x.astype('f4'), centroids[key][c].astype('f4'))
                min_dists_category.append(d)
            min_dists.append(np.min(min_dists_category))
        y_pred.append(np.argmin(min_dists))
        cnt += 1
    return y_pred


def y_pred_with_fgn_counts(x_features, centroids, metric, thr):
    # TODO number of centroids is radically different for each category
    # I need to find a weighted measurement that takes into account
    # number of samples per cluster etc.

    y_pred = []

    cnt = 0
    for x in x_features:
        if cnt % 5000 == 0:
            print(cnt)
        max_counts = []
        # each class
        for key in centroids:
            counts = 0
            # each centroid of each class
            # print(key, len(centroids[key]))
            for cent in centroids[key]:
                # print(cent.shape)
                # set_trace()
                d = metric.dist(x.astype('f4'), cent.astype('f4'))
                if d <= thr:
                    counts += 1
            max_counts.append(counts)
        # print(max_counts)
        # set_trace()
        y_pred.append(np.argmax(max_counts))
        cnt += 1
    return y_pred


if __name__ == "__main__":

    # TODO focus on FGM with merge! So that you can speed up the samples.
    # centroids, y_train_pred = fsn_training_with_fgn(x_train_part, y_train_part,
    #                                                 thresholds, metric)
    # centroids, y_train_pred = fsn_training_with_fgn_ma(x_train_part,
    #                                                    y_train_part,
    #                                                    thresholds,
    #                                                    metric)
    centroids, y_train_pred = fsn_training_with_fgn_shuffles(x_train_part,
                                                             y_train_part,
                                                             thresholds,
                                                             metric)
    # centroids, y_train_pred = fsn_training_no_fgn(x_train_part, y_train_part,
    #                                               metric)

    # y_test_pred = fsn_test_with_fgn(x_test_part, centroids, metric)
    # y_test_pred = fsn_test_no_fgn(x_test_part, centroids, metric)

    print('Accuracy training set',
          report_accuracy(y_train_part, y_train_pred))
    # print('Accuracy test set (min_dist)',
    #       report_accuracy(y_test_part, y_test_pred))

    # for t in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.2]:
    #     y_test_pred2 = y_pred_with_fgn_counts(x_test_part, centroids, metric, thr=t)
    #     print('Accuracy test set (max_counts)', t, report_accuracy(y_test_part, y_test_pred2))

    # print('Exactness training set', report_exactness(y_train_part, y_train_pred))
    # print('Exactness test set', report_exactness(y_test, y_test_pred))

    # for i in centroids:
    #    figure(); imshow(centroids[i])

    # thresholds = [0.1, 0.08, 0.05]
    # all_levels = fgn_mf(x_train_part, thresholds=thresholds,
    #                     metric=metric, verbose=True)

    # for i in range(len(thresholds)):
    #     print(len(all_levels[i + 1].centroids))

    # print('Number of centroids found', len(mc_map.centroids))
