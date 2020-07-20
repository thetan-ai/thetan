from robotikos.fgm import assign_to_centroids, fgn_shuffles
from robotikos.metric import (NormalizedSumAbsDiffMetric,
                              NormalizedSumSquaredDiffMetric)
from robotikos.metric import IdentityFeature
from robotikos.metricspeed import (assign_to_centroid_nl1,
                                   assign_to_centroid_nl2)
from robotikos.rsvd import bksvd
from sklearn.metrics import accuracy_score
from time import time


import numpy as np
import tensorflow as tf


def fsn_test_with_fgn_shuffles(x_test, centroids, metric):
    return y_pred_with_fgn_shuffles_fast(x_test, centroids, metric)


def fsn_training_with_fgn_shuffles(x_train, y_train, thresholds, metric,
                                   num_shuffles=1, mode='fgn_ma',
                                   verbose=False, level=None, num_labels=10):
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
                                num_shuffles=num_shuffles, mode=mode,
                                verbose=verbose)
        centroids[s] = collect2[level]['centroids']
    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])
    if verbose:
        print('Total number of centroids', total_no_centroids)
    y_pred = y_pred_with_fgn_shuffles_fast(x_train, centroids, metric)
    return centroids, y_pred


def report_accuracy(y_true, y_pred):
    accs = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    return accs


def separate_train_data(x_train, y_train, num_labels=10):
    """ Separates training data to each category"""
    sep_data = {}
    for i in range(num_labels):
        y_arg = np.where(y_train == i)
        sep_data[i] = x_train[y_arg[0]]
    return sep_data


def y_pred_with_fgn_shuffles_fast(x_features, centroids, metric):
    cnt = 0
    lengths = [len(centroids[key]) for key in centroids]
    total_centroids = np.sum(lengths)
    feature_shape = x_features[0].shape
    all_centroids = np.zeros((total_centroids,) + feature_shape)
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
    labels = assign_to_centroid_fast(metric, x_features.astype('f4'),
                                     all_centroids.astype('f4'))
    tiled = np.repeat(np.arange(len(centroids.keys())), tuple(lengths))
    y_pred = [tiled[l] for l in labels]
    y_pred = np.array(y_pred)
    return y_pred


if __name__ == '__main__':
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    #metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())

    #thresholds = [0.15, 0.1, 0.08]
    thresholds = [.1, .08, .004]

    print('Thresholds', thresholds)

    num_shuffles = 10
    train_sz = y_train.shape[0]
    test_sz = y_test.shape[0]
    use_sobel = False
    use_bksvd = False
    k = 10
    verbose = True
    level = None

    print('Training size', train_sz)
    print('Test size', test_sz)

    data_shape = x_train.shape[1:]
    # [N, 1, 3072=(32x32x3)]
    x_train_rsh = np.reshape(
        x_train, (y_train.shape[0], 1, np.prod(data_shape)))
    # [N, 3, 1024=(32x32)]

    x_train_part = x_train_rsh  # x_train.copy()[:train_sz]
    x_train_part = np.ascontiguousarray(x_train_part / 255.)
    # low rank approximation
    x_train_part_ev = np.zeros((x_train_part.shape[0], 1, k))

    if use_bksvd:
        t1 = time()
        for i in range(train_sz):
            x_train_part_ev[i][0] = bksvd(x_train_part[i], k=k)[1]
        t2 = time()
        print('Duration bksvd %.3f s' % (t2 - t1,))
        x_train_part = x_train_part_ev

    # [N, 1, 3072=(32x32x3)]
    x_test_rsh = np.reshape(
        x_test, (y_test.shape[0], 1, np.prod(data_shape)))
    # [N, 3, 1024=(32x32)]

    x_test_part = x_test_rsh  # x_test.copy()[:test_sz]
    x_test_part = np.ascontiguousarray(x_test_part / 255.)
    x_test_part_ev = np.zeros((x_test_part.shape[0], 1, k))

    if use_bksvd:
        for i in range(test_sz):
            x_test_part_ev[i][0] = bksvd(x_test_part[i], k=k)[1]
        x_test_part = x_test_part_ev

    y_train_part = y_train.copy()[:train_sz]
    y_test_part = y_test.copy()[:test_sz]

    feature_shape = x_train_part[0].shape

    t1 = time()
    centroids, y_train_pred = fsn_training_with_fgn_shuffles(
        x_train_part, y_train_part, thresholds, metric,
        num_shuffles=num_shuffles, mode='fgn_ma',
        verbose=verbose, level=level, num_labels=10)
    t2 = time()
    print('Duration FGN shuffles training %.3f s' % (t2 - t1,))

    total_centroids = np.sum([len(centroids[c]) for c in centroids])
    print('Total number of centroids ', total_centroids)

    t1 = time()
    y_test_pred = fsn_test_with_fgn_shuffles(x_test_part, centroids, metric)
    t2 = time()
    print('Duration FGN shuffles testing %.3f s' % (t2 - t1,))

    print('Accuracy training set', report_accuracy(y_train_part, y_train_pred))
    print('Accuracy test set (min_dist)', report_accuracy(y_test_part,
                                                          y_test_pred))
