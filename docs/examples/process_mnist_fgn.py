from pdb import set_trace
from time import time
import numpy as np
from skimage import filters
from robotikos.clustering import FastGroupingNode, ClusterMapCentroid, ClusterCentroid
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from scipy.linalg import qr, svd
from robotikos.fgm import fgn_ma, fgn_shuffles, assign_to_centroids
from robotikos.rsvd import bksvd
from robotikos.metricspeed import (assign_to_centroid_nl1,
                                   assign_to_centroid_nl2)
from robotikos.fsn import fsn_training_with_fgn_shuffles
from robotikos.fsn import fsn_test_with_fgn_shuffles


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

    dataset_type = 'fashion'

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
    # thresholds = [0.5, 0.2, 0.08, 0.05]  # Provided 95% accuracy for 1 shuffle and 0.9652 for 10.
    # thresholds = [0.5, 0.2, 0.08]  # 80% acc (1 shuffle), 85% acc (10) and 85.5% acc (50)
    # thresholds = [0.5, 0.2, 0.08, 0.07, 0.06, 0.05]
    # thresholds = [0.5, 0.2, 0.08, 0.06, 0.04]
    # thresholds = [0.07, 0.06, 0.05, 0.04]
    thresholds = [0.1, 0.08]
    thresholds = [0.5, 0.25, 0.1, 0.05]  # This works well for MNIST for L1 metric 0.966
    # L1 1 shuffle (many centroids)
    # Accuracy training set 0.9949333333333333
    # Accuracy test set (min_dist) 0.9626
    thresholds = [0.15, 0.1]
    thresholds = [0.15, 0.1, 0.08]
    # L1 1 shuffle
    # Total number of centroids 6402
    # Duration FGN shuffles training 774.688 s
    # Total number of centroids  6402
    # Duration FGN shuffles testing 100.918 s
    # Accuracy training set 0.9684833333333334
    # Accuracy test set 0.9452
    # L1 10 shuffles
    # Total number of centroids 8587
    # Duration FGN shuffles training 1272.338 s
    # Total number of centroids  8587
    # Duration FGN shuffles testing 135.132 s
    # Accuracy training set 0.98245
    # Accuracy test set (min_dist) 0.9536
    thresholds = [0.1, 0.075, 0.05, 0.025]  # fashion mnist 0.93 (train acc.) and 0.82 (test acc.)
    thresholds = [0.1, 0.075, 0.05]  # fashion mnist threshold
    thresholds = [0.1, 0.075]  # fashion mnist threshold (binary 0.2) 0.86 (train acc.) and 0.80 (test acc.)
    print('Thresholds', thresholds)

    num_shuffles = 10
    train_sz = 60000
    test_sz = 10000
    use_sobel = False
    use_bksvd = False
    use_binary = True
    k = 10
    verbose = True
    level = None

    print('Training size', train_sz)
    print('Test size', test_sz)

    x_train_part = x_train.copy()[:train_sz]
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

    if use_binary:
        x_train_part[x_train_part > 0.2] = 1.
        x_train_part[x_train_part <= 0.2] = 0.

    import pylab as pl

    # for i in range(10):
    #     pl.figure()
    #     pl.imshow(x_train_part[i])

    # pl.show()

    x_test_part = x_test.copy()[:test_sz]
    x_test_part = np.ascontiguousarray(x_test_part / 255.)
    x_test_part_ev = np.zeros((x_test_part.shape[0], 1, k))

    if use_bksvd:
        for i in range(test_sz):
            x_test_part_ev[i][0] = bksvd(x_test_part[i], k=k)[1]
        x_test_part = x_test_part_ev

    if use_binary:
        x_test_part[x_test_part > 0.2] = 1.
        x_test_part[x_test_part <= 0.2] = 0.

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

    print('Accuracy training set',
          report_accuracy(y_train_part, y_train_pred))
    print('Accuracy test set (min_dist)',
          report_accuracy(y_test_part, y_test_pred))

    pass

    # for t in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.2]:
    #     y_test_pred2 = y_pred_with_fgn_counts(x_test_part, centroids, metric, thr=t)
    #     print('Accuracy test set (max_counts)', t, report_accuracy(y_test_part, y_test_pred2))

    # print('Exactness training set', report_exactness(y_train_part, y_train_pred))
    # print('Exactness test set', report_exactness(y_test, y_test_pred))

