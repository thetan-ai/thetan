from time import time
import numpy as np
from numpy import testing as npt
from robotikos.theta import theta_g1
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
    summs, clusters_sizes, centroids, indices = theta_g1(X, theta=theta, metric='euclidean')
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
    summs, clusters_sizes, centroids, indices = theta_g1(X, theta=theta, metric='norm_ssd')
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

test_theta_g1_large_array(theta=1.0, size=(500000, 300))