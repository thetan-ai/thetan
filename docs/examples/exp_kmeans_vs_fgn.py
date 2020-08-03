from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.datasets import make_blobs
import pylab as pl
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from robotikos.fgm import fgn_shuffles, fgn_ma
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from sklearn.cluster import KMeans
from robotikos.metricspeed import distance_matrix


def evaluate_centroids(ecentroids, centroids, thr=0.5):
    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    D = np.sqrt(distance_matrix(metric,
                                np.ascontiguousarray(ecentroids[:, None, :]),
                                np.ascontiguousarray(centroids[:, None, :])))

    closest = np.min(D, axis=1)
    count = np.sum(closest < thr)
    print('>>>>> ', count)
    return count


def plot_centroids_clusters(data, centroids, labels, title, initial_centroids=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(len(points)):
        new_colors[ind] = cluster_colors[labels[ind]]

    pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')
    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')


exp_type = 'disks'
methods = ['kmeans', 'fgn_ma', 'birch', 'mb_kmeans', 'gmm']
# methods = ['kmeans', 'fgn_ma']
evaluation_thr = 0.3
verbose = True

misses = {}
timings = {}

if exp_type == 'blobs':
    # Generate centers for the blobs so that it forms a 10 X 10 grid.
    # xx = np.linspace(-22, 22, 10)
    # yy = np.linspace(-22, 22, 10)
    xx = np.linspace(-44, 44, 20)
    yy = np.linspace(-44, 44, 20)
    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                           np.ravel(yy)[:, np.newaxis]))

    X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)

    points = X
    sz = len(n_centres)
    centroids = np.array(n_centres, dtype='f4')


if exp_type == 'disks':

    np.random.seed(0)
    sz = (50, 6)

    coords = []
    centroids = []

    for x, y in np.ndindex(sz):

        for d in range(100):
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = 5 * x + r * np.cos(theta)
            # y1 = 5 * y + 2 * r * np.sin(theta)
            y1 = 5 * y + r * np.sin(theta)
            coords.append([x1, y1])
        centroids.append([5 * x, 5 * y])

    points = np.array(coords)
    centroids = np.array(centroids, dtype='f4')


print('Before shuffling')
print(points[:10])

np.random.shuffle(points)

print('After shuffling')
print(points[:10])

print('Number of gold centroids ', np.prod(sz))

if 'kmeans' in methods:

    t1 = time()
    kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=10,
                    max_iter=300, tol=0.0001, random_state=0).fit(points)
    t2 = time()

    kmeans_time = t2 - t1
    print('kmeans++ ', kmeans_time)

    kcentroids = kmeans.cluster_centers_
    klabels = kmeans.labels_
    if verbose:
        plot_centroids_clusters(points, kcentroids, klabels,
                                'kmeans++',
                                initial_centroids=centroids)

    misses['kmeans++'] = np.abs(len(centroids) - evaluate_centroids(kcentroids, centroids, evaluation_thr))
    timings['kmeans++'] = kmeans_time

    tmp = 0

if 'fgn_ma' in methods:

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

    elbow_check = []
    theta_range = [5.365]  # np.linspace(3.8, 5.5, num=40) # np.linspace(1, 3.5, num=50)
    theta_range = [7.25]  # with blobs 5.8 (104), 6.0 (101), 6.2 (102), 6.4 (101), 6.6 (102), 7.0 (101), 7.2 (101), 7.3 (99), 7.5 (98), 8.0 (96)
    theta_range = [6.0]
    # theta_range = [20., 10, 5.5]

    for fgn_t in np.linspace(5.5, 6.5, num=10):
        t1 = time()
        res = fgn_shuffles(points[:, None, :], thresholds=[fgn_t],
                           metric=metric,
                           num_shuffles=1, mode='fgn_ma', verbose=True)

        # res = fgn_ma(points[:, None, :], thresholds=theta_range, metric=metric,
        #              select_randomly=None, rng=None, verbose=True)

        t2 = time()

        fgn_time = t2 - t1
        print('fgn ', fgn_time)
        print('Speedup', kmeans_time/fgn_time)

        level = sorted(res.keys())[-1]
        try:
            fcentroids = res[level]['centroids']
            flabels = res[level]['labels']
        except TypeError:
            fcentroids_ = res[level].centroids
            fcentroids = {}
            for c in range(len(fcentroids_)):
                fcentroids[c] = fcentroids_[c]

            flabels = np.zeros(len(points), dtype='i8')
            for c in range(len(fcentroids_)):
                indices = res[level][c].indices
                flabels[indices] = c

        print('# centroids', len(fcentroids.keys()))
        elbow_check.append(len(fcentroids.keys()))

        fgn_centroids = np.zeros((len(fcentroids), 2))
        for i, c in enumerate(fcentroids.keys()):
            fgn_centroids[i] = np.squeeze(fcentroids[c])

        ext = str(np.round(fgn_t, 2))
        if verbose:
            plot_centroids_clusters(points, fgn_centroids, flabels, 'fgn_ma ' + ext)

        misses['fgn_ma ' + ext] = np.abs(len(centroids) - evaluate_centroids(fgn_centroids, centroids, evaluation_thr))
        timings['fgn_ma ' + ext] = fgn_time


if 'birch' in methods:

    for bt in np.linspace(1.5, 2.5, 10):

        t1 = time()
        # birch = Birch(threshold=1.7, n_clusters=None).fit(points)
        # birch = Birch(threshold=1.7, n_clusters=int(np.prod(sz))).fit(points)
        # birch = Birch(threshold=3.0, n_clusters=int(np.prod(sz))).fit(points)
        birch = Birch(threshold=bt, n_clusters=int(np.prod(sz))).fit(points)

        t2 = time()

        birch_time = t2 - t1
        print('birch_time ', birch_time)

        birch_labels = birch.labels_
        birch_centroids = birch.subcluster_centers_
        print('# centroids ', len(birch_centroids))
        if verbose:
            plot_centroids_clusters(points, birch_centroids, birch_labels, 'birch ' + str(np.round(bt, 2)))

        misses['birch ' + str(np.round(bt, 2))] = np.abs(len(centroids) - evaluate_centroids(birch_centroids, centroids, evaluation_thr))
        timings['birch ' + str(np.round(bt, 2))] = birch_time

if 'mb_kmeans' in methods:

    t1 = time()
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=int(np.prod(sz)),
                          batch_size=100,
                          n_init=10, max_no_improvement=10, verbose=0,
                          random_state=0).fit(points)
    t2 = time()
    mb_kmeans_time = t2 - t1
    print('mb_kmeans_time ', mb_kmeans_time)

    mbk_labels = mbk.labels_
    mbk_centroids = mbk.cluster_centers_
    print('# centroids ', len(mbk_centroids))
    if verbose:
        plot_centroids_clusters(points, mbk_centroids, mbk_labels, 'mb_kmeans')

    misses['mb_kmeans'] = np.abs(len(centroids) - evaluate_centroids(mbk_centroids, centroids, evaluation_thr))
    timings['mb_kmeans'] = mb_kmeans_time

if 'gmm' in methods:

    t1 = time()
    cov_type = ['spherical', 'diag', 'tied', 'full']
    gmm = GaussianMixture(n_components=int(np.prod(sz)),
                          covariance_type=cov_type[3],
                          max_iter=200, random_state=0).fit(points)
    t2 = time()
    gmm_time = t2 - t1
    print('# gmm_time ', gmm_time)

    gmm_centroids = gmm.means_
    gmm_labels = gmm.predict(points)

    if verbose:
        plot_centroids_clusters(points, gmm_centroids, gmm_labels, 'gmm')

    misses['gmm'] = np.abs(len(centroids) - evaluate_centroids(gmm_centroids, centroids, evaluation_thr))
    timings['gmm'] = gmm_time

# pl.figure()
# pl.plot(theta_range, elbow_check)
# pl.grid()

pl.figure()
pl.title('Misses')
x_tmp = np.arange(len(misses.values()))
pl.bar(x_tmp, misses.values())
pl.xticks(x_tmp, misses.keys())

pl.figure()
pl.title('Timings')
x_tmp = np.arange(len(timings.values()))
pl.bar(x_tmp, timings.values())
pl.xticks(x_tmp, timings.keys())


pl.show()


t = 1

