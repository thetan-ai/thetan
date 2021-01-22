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
                              NormalizedSumAbsDiffMetric,
                              NormalizedSumSquaredRootDiffMetric)
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, rand_score, adjusted_rand_score
from robotikos.metricspeed import distance_matrix


def evaluate_centroids(ecentroids=None, centroids=None, thr=0.5, mode='NMI', X=None, label=None, true_label=None):
    if mode == 'SSD':
        metric = NormalizedSumSquaredRootDiffMetric(feature=IdentityFeature())
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

modes = ['NMI', 'SIL', 'FMS', 'SSD', 'RAI']
evaluation_thr = 0.1
verbose = True
show_thr = False
radius = 4
n_local_samples = 10
blob_dis = 33
cluster_per_dim = 10
scores = {}
timings = {}



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

points_idx = np.arange(np.shape(points)[0])
np.random.shuffle(points_idx)
new_points = np.zeros(np.shape(points))
new_labels = np.zeros(np.shape(labels))
for i, idx in zip(range(np.shape(points)[0]), points_idx):
    new_points[i] = points[idx]
    new_labels[i] = labels[idx]

points = np.ascontiguousarray(new_points)
labels = np.ascontiguousarray(new_labels)

print('After shuffling')
print(points[:10])

print('Number of points ', points.shape)

print('Number of gold centroids ', np.prod(sz))

theta = 3.0

metric = NormalizedSumSquaredRootDiffMetric(feature=IdentityFeature())
centroids_list = []
for idx in range(10):
    res = fgn_shuffles(points[:, None, :], thresholds=[theta],
                        metric=metric,
                        num_shuffles=1, mode='fgn', verbose=False)



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


    fgn_centroids = np.zeros((len(fcentroids), 2))
    for i, c in enumerate(fcentroids.keys()):
        fgn_centroids[i] = np.squeeze(fcentroids[c])
    for c in fgn_centroids:
        centroids_list.append(c)
    if idx == 0:
        plot_centroids_clusters(points, fgn_centroids, flabels, 'fgn ' +str(idx), initial_centroids=centroids)
        scores['fgn ' +str(idx)] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['fgn ' +str(idx)][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=flabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['fgn ' +str(idx)][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=flabels, true_label=labels)
            elif mode == 'SSD':
                scores['fgn ' +str(idx)][mode] = evaluate_centroids(fgn_centroids, centroids, evaluation_thr, mode=mode)
        
    else:
        res = fgn_shuffles(np.ascontiguousarray(centroids_list)[:, None, :], thresholds=[theta],
                                                metric=metric,
                                                num_shuffles=1, mode='fgn, verbose=False')
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

        fgn_centroids = np.zeros((len(fcentroids), 2))
        for i, c in enumerate(fcentroids.keys()):
            fgn_centroids[i] = np.squeeze(fcentroids[c])
        scores['fgn ' + str(idx)] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['fgn ' +str(idx)][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=flabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['fgn ' +str(idx)][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=flabels, true_label=labels)
            elif mode == 'SSD':
                scores['fgn ' +str(idx)][mode] = evaluate_centroids(fgn_centroids, centroids, evaluation_thr, mode=mode)



pl.figure()
pl.title('Score')
x_tmp = np.arange(len(modes))
width = 1/len(scores.keys())

for i, method in zip(range(len(scores.keys())), scores.keys()):
    pl.bar(x_tmp + width * i, scores[method].values(), width, label=method)
pl.xticks(x_tmp + width / 2 + (len(scores.keys())-2) * width/2 , modes)
pl.legend()

pl.show()

