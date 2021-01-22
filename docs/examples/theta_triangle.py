from time import time
import numpy as np
from numpy import testing as npt
from robotikos.theta import theta_g1, assign
from robotikos.fgm import fgn_shuffles, fgn
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

# summs, clusters_sizes, indices = theta_g1(X, theta=1000.0)


# print(X.mean(axis=0))
# print(summs)
# print(clusters_sizes)
# print(summs/clusters_sizes.astype('f8'))

def test_theta_g1(verbose=False):
    X = np.random.normal(size=(500000, 2))

    X[:250000] += np.array([20, 0.])

    # np.random.shuffle(X)

    X = np.ascontiguousarray(X, dtype='f8')

    # X = X[:10].copy()
    # print(X)

    theta = 10.0

    t1 = time()    
    centroids, indices, summs, clusters_sizes = theta_g1(X, theta=theta, metric='euclidean', all_out=True)
    t2 = time()
    tg1_duration = t2 - t1
    print('TG1 time', t2-t1)
    print('sizes', clusters_sizes)
    print('centroids', centroids)
    print('size of indices', len(indices))


    X2 = X.copy()[:, None, :]

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    t1 = time()
    all_levels = fgn(X2, thresholds=[theta*theta], metric=metric)
    t2 = time()
    fgn_duration = t2 - t1
    print('FGN time', fgn_duration)
    print('centroids', all_levels[1].centroids)
    print('Speedup ', fgn_duration/tg1_duration)


    fgn_centroids = np.asarray(all_levels[1].centroids)

    if verbose:
        from pylab import figure, plot, show , axis

        figure()
        plot(X[:, 0], X[:, 1], '.')
        plot(centroids[:, 0], centroids[:, 1], 'rx')
        plot(fgn_centroids[:, 0, 0], fgn_centroids[:, 0, 1], 'bo')
        axis('equal')
        show()

    npt.assert_array_almost_equal(centroids, np.squeeze(fgn_centroids), decimal=3)

# test_theta_g1()

def test_theta_g1_large_array(theta=10.0, size=(500000, 200), verbose=False):
    
    X = np.random.normal(size=size)

    #X[:250000] += np.array([20, 0.])

    # np.random.shuffle(X)

    # X = np.ascontiguousarray(X, dtype='f8')

    # X = X[:10].copy()
    # print(X)

    theta = theta

    t1 = time()
    centroids, indices, summs, clusters_sizes = theta_g1(X, theta=theta, metric='norm_ssd', all_out=True)
    t2 = time()
    tg1_duration = t2 - t1
    print('TG1 time', t2-t1)
    # print('sizes', clusters_sizes)
    # print('centroids', centroids)
    # print('size of indices', len(indices))
    print('no clusters', centroids.shape[0])

    X2 = X.copy()[:, None, :]

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    t1 = time()
    all_levels = fgn(X2, thresholds=[theta], metric=metric)
    t2 = time()
    fgn_duration = t2 - t1
    print('FGN time', fgn_duration)
    # print('centroids', all_levels[1].centroids)
    print('Speedup ', fgn_duration/tg1_duration)

    fgn_centroids = np.asarray(all_levels[1].centroids)

    print('no clusters', fgn_centroids.shape[0])

    verbose = False
    if verbose:

        from pylab import figure, plot, show , axis

        figure()
        plot(X[:, 0], X[:, 1], '.')
        plot(centroids[:, 0], centroids[:, 1], 'rx')
        plot(fgn_centroids[:, 0, 0], fgn_centroids[:, 0, 1], 'bo')
        axis('equal')
        show()

    # npt.assert_array_almost_equal(centroids, np.squeeze(fgn_centroids), decimal=3)
    pass

# test_theta_g1_large_array(theta=1.6, size=(500000, 300))
# test_theta_g1_large_array(theta=1.6, size=(500000, 30))
# test_theta_g1_large_array(theta=1.6, size=(500000, 10))


def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)
    
    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids


def theta_g2(X, theta, metric='euclidean', num_shuffles=10):

    all_centroids = {}
    X_work = X.copy()

    num_centroids = []
    timings = []
    for s in range(num_shuffles):

        np.random.shuffle(X_work)
        X_work_cp = np.ascontiguousarray(X_work)
        t1 = time()
        centroids, _ = theta_g1(X_work_cp, theta=theta, metric=metric)
        t2 = time()        
        timings.append(t2-t1)

        all_centroids[s] = centroids
        num_centroids.append(len(centroids))

    print('Time for tg1 calls', np.sum(timings))
    total_num_centroids = np.sum(num_centroids)
    X_centroids = np.zeros((total_num_centroids, X_work.shape[1]), dtype='f8')

    offset = 0

    for s in range(num_shuffles):

        X_centroids[offset:offset + all_centroids[s].shape[0]] = all_centroids[s]
        offset += all_centroids[s].shape[0]
    
    t1 = time()
    new_centroids, _ = theta_g1(X_centroids, theta=theta, metric=metric)
    t2 = time()
    print('Time for tg1 on centroids', t2 - t1)

    t1 = time()
    final_indices = assign(X, new_centroids, metric=metric)
    t2 = time()
    print('Time for assign calls', t2 - t1)
    final_centroids = labels_to_centroids(X, final_indices)

    return final_centroids, final_indices


def test_theta_g2(verbose=False):
    X = np.random.normal(size=(500000, 2))

    X[:250000] += np.array([5, 0.])
    # X[:250000] += np.array([10, 0.])

    # np.random.shuffle(X)

    X = np.ascontiguousarray(X, dtype='f8')

    # X = X[:10].copy()
    # print(X)

    theta = 10.0
    theta = 5.0

    t1 = time()    
    # centroids, indices = theta_g1(X, theta=theta*theta, metric='norm_ssd')
    centroids, indices = theta_g2(X, theta=theta*theta, metric='norm_ssd', num_shuffles=10)
    t2 = time()
    tg2_duration = t2 - t1
    print('TG2 time', tg2_duration)
    print('centroids', centroids)
    print('size of indices', len(indices))


    X2 = X.copy()[:, None, :]

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    t1 = time()
    all_levels = fgn_shuffles(X2, thresholds=[theta*theta], metric=metric, num_shuffles=10)
    t2 = time()
    fgn_duration = t2 - t1
    print('FGN time', fgn_duration)
    print('centroids', all_levels[1]['centroids'])
    print('Speedup ', fgn_duration/tg2_duration)

    if verbose:
        from pylab import figure, plot, show , axis

        figure()
        plot(X[:, 0], X[:, 1], '.')
        plot(centroids[:, 0], centroids[:, 1], 'rx')
        for k in all_levels[1]['centroids'].keys():
            fgn_centroid = all_levels[1]['centroids'][k]
            plot(fgn_centroid[0, 0], fgn_centroid[0, 1], 'bo')
        axis('equal')
        show()


test_theta_g2(True)
