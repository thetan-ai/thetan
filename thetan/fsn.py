import numpy as np
from robotikos.fgm import fgn_ma, fgn_shuffles, assign_to_centroids
from robotikos.rsvd import bksvd
from robotikos.clustering import FastGroupingNode, ClusterMapCentroid, ClusterCentroid
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
import tensorflow as tf
from robotikos.metricspeed import (assign_to_centroid_nl1,
                                   assign_to_centroid_nl2)


def separate_train_data(x_train, y_train, num_labels=10):
    """ Separates training data to each category"""

    sep_data = {}
    for i in range(num_labels):
        sep_data[i] = x_train[y_train == i]

    return sep_data


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
        all_levels = fgn_ma(sep_data[s], thresholds, metric)
        allk = sorted(all_levels.keys())
        centroids[s] = all_levels[allk[-1]].centroids

    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])
    print('Total number of centroids', total_no_centroids)

    y_pred = y_pred_with_fgn(x_train, centroids, metric)

    return centroids, y_pred


def fsn_training_with_fgn_shuffles(x_train, y_train, thresholds,
                                   metric, num_shuffles=1,
                                   mode='fgn_ma',
                                   verbose=False, level=None,
                                   num_labels=10):

    sep_data = separate_train_data(x_train, y_train, num_labels=num_labels)

    if verbose:
        print('#')
        print('FGN SHUFFLES version')
        print('Thresholds', thresholds)

    if level is None:
        level = len(thresholds)

    centroids = {}
    for s in sep_data:

        print('>> Processing label ', s)
        collect2 = fgn_shuffles(sep_data[s], thresholds, metric=metric,
                                num_shuffles=num_shuffles,
                                mode=mode, verbose=verbose)
        centroids[s] = collect2[level]['centroids']

    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])

    if verbose:
        print('Total number of centroids', total_no_centroids)

    y_pred = y_pred_with_fgn_shuffles_fast(x_train, centroids, metric)

    return centroids, y_pred


def fsn_test_with_fgn(x_test, centroids, metric):
    return y_pred_with_fgn(x_test, centroids, metric)


def fsn_test_with_fgn_shuffles(x_test, centroids, metric):
    return y_pred_with_fgn_shuffles_fast(x_test, centroids, metric)


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


def y_pred_with_fgn_shuffles_fast(x_features, centroids, metric):
    y_pred = []
    cnt = 0
    lengths = [len(centroids[key]) for key in centroids]
    total_centroids = np.sum(lengths)
    feature_shape = x_features[0].shape
    all_centroids = np.zeros((total_centroids,
                              feature_shape[0],
                              feature_shape[1]))
    for key in centroids:
        for c in centroids[key]:
            all_centroids[cnt] = centroids[key][c]
            cnt += 1

    if isinstance(metric, NormalizedSumSquaredDiffMetric):
        assign_to_centroid_fast = assign_to_centroid_nl2
    elif isinstance(metric, NormalizedSumAbsDiffMetric):
        assign_to_centroid_fast = assign_to_centroid_nl1
    else:
        print('Unknown metric to use here - switching to slow version')
        assign_to_centroid_fast = assign_to_centroids

    labels = assign_to_centroid_fast(metric,
                                     x_features.astype('f4'),
                                     all_centroids.astype('f4'))
    tiled = np.repeat(np.arange(len(centroids.keys())), tuple(lengths))

    y_pred = [tiled[l] for l in labels]
    y_pred = np.array(y_pred)

    return y_pred
