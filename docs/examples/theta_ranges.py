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
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, rand_score, adjusted_rand_score
from robotikos.metricspeed import distance_matrix


def evaluate_centroids(ecentroids=None, centroids=None, thr=0.5, mode='NMI', X=None, label=None, true_label=None):
    if mode == 'SSD':
        metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
        D = np.sqrt(distance_matrix(metric,
                                    np.ascontiguousarray(ecentroids[:, None, :]),
                                    np.ascontiguousarray(centroids[:, None, :])))

        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)
        score = count / len(centroids)

    elif mode == 'NMI':
        score = normalized_mutual_info_score(true_label, label)

    elif mode == 'SIL':
        score = silhouette_score(X, label)

    elif mode == 'FMS':
        score = fowlkes_mallows_score(true_label, label)

    elif mode == 'RAI':
        score = adjusted_rand_score(true_label, label)
        #score = rand_score(true_label, label)

    else:
        print('no mode')
        score = 0

    print('>>>>> ', score)
    return score


def plot_centroids_clusters(data, centroids, labels, title, initial_centroids=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(len(points)):
        new_colors[ind] = cluster_colors[labels[ind]]

    pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')
    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')
    pl.show()


exp_type = 'easy'
methods = ['fgn_ma']
modes = ['NMI', 'SIL', 'FMS', 'SSD', 'RAI']
evaluation_thr = 0.1
verbose = False
show_thr = False
radius = 4
n_local_samples = 100
blob_dis = 33
cluster_per_dim = 10
scores = {}
timings = {}

if exp_type == 'easy':
    # Generate centers for the blobs so that it forms a 10 X 10 grid.
    xx = np.linspace(-22, 22, 5)
    yy = np.linspace(-22, 22, 5)
    # blob_dis = 10
    # cluster_x = 100
    # cluster_y = 100
    # xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
    # yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                           np.ravel(yy)[:, np.newaxis]))

    # X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
    # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], 2))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
    for i in range(n_centres.shape[0]):
        cpos = n_centres[i]
        # local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
        local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    gt_centroids = np.array(gt_centroids)

    points = X
    labels = y
    sz = len(n_centres)
    # centroids = np.array(n_centres, dtype='f4')
    centroids = gt_centroids

print('Before shuffling')
print(points[:10])

points_idx = np.arange(np.shape(points)[0])
np.random.shuffle(points_idx)
new_points = np.zeros(np.shape(points))
new_labels = np.zeros(np.shape(labels))
for i, idx in zip(range(np.shape(points)[0]), points_idx):
    new_points[i] = points[idx]
    new_labels[i] = labels[idx]

points = new_points
labels = new_labels

print('After shuffling')
print(points[:10])

print('Number of points ', points.shape)
print('Number of gold centroids ', np.prod(sz))


if 'fgn_ma' in methods:

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

    elbow_check = []
    theta_range = [5.365]  # np.linspace(3.8, 5.5, num=40) # np.linspace(1, 3.5, num=50)
    theta_range = [7.25]  # with blobs 5.8 (104), 6.0 (101), 6.2 (102), 6.4 (101), 6.6 (102), 7.0 (101), 7.2 (101), 7.3 (99), 7.5 (98), 8.0 (96)
    theta_range = [6.0]
    # theta_range = [20., 10, 5.5]
    theta_range = np.arange(0., 500, 1)
    # theta_range = [25]

    number_of_clusters = []

    for fgn_t in theta_range: #theta_range: #np.linspace(5., 5.5, num=5):
        t1 = time()
        res = fgn_shuffles(points[:, None, :], thresholds=[fgn_t],
                           metric=metric,
                           num_shuffles=1, mode='fgn', verbose=True)

        # res = fgn_ma(points[:, None, :], thresholds=theta_range, metric=metric,
        #              select_randomly=None, rng=None, verbose=True)

        t2 = time()

        fgn_time = t2 - t1
        print('fgn ', fgn_time)
        # print('Speedup', kmeans_time/fgn_time)

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

        number_of_clusters.append(len(fgn_centroids))

        ext = str(np.round(fgn_t, 2))
        if not show_thr:
            ext = ''
        if verbose:
            plot_centroids_clusters(points, fgn_centroids, flabels, 'fgn ' + ext, initial_centroids=centroids)
        timings['fgn ' + ext] = fgn_time

    pl.figure()
    pl.grid()
    number_of_clusters = np.array(number_of_clusters)
    indices = np.where(number_of_clusters == 25)[0]
    print('25 p=', len(indices)/len(number_of_clusters))
    pl.plot(theta_range, number_of_clusters, '.')
    pl.plot(theta_range[indices], number_of_clusters[indices], 'g.')
    indices = np.where(number_of_clusters == 4)[0]
    print('4 p=', len(indices)/len(number_of_clusters))
    pl.plot(theta_range[indices], number_of_clusters[indices], 'm.')
    indices = np.where(number_of_clusters == 1)[0]
    print('1 p=', len(indices)/len(number_of_clusters))
    pl.plot(theta_range[indices], number_of_clusters[indices], 'r.')
    pl.show()

tmp = 1
