from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.datasets import make_blobs
import pylab as pl


exp_type = 'blobs'

if exp_type == 'blobs':
    # Generate centers for the blobs so that it forms a 10 X 10 grid.
    xx = np.linspace(-22, 22, 10)
    yy = np.linspace(-22, 22, 10)
    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                           np.ravel(yy)[:, np.newaxis]))

    X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)

    points = X
    sz = len(n_centres)


if exp_type == 'disks':

    np.random.seed(0)
    sz = (50, 3)

    coords = []
    centroids = []

    for x, y in np.ndindex(sz):

        for d in range(100):
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = 5 * x + r * np.cos(theta)
            y1 = 5 * y + 2 * r * np.sin(theta)
            coords.append([x1, y1])
        centroids.append([x, y])

    points = np.array(coords)

print('Before shuffling')
print(points[:10])

np.random.shuffle(points)

print('After shuffling')
print(points[:10])

print('Number of centroids ', np.prod(sz))

from sklearn.cluster import KMeans

t1 = time()
kmeans = KMeans(n_clusters=np.prod(sz), random_state=0).fit(points)
t2 = time()

kmeans_time = t2 - t1
print('kmeans++ ', kmeans_time)

kcentroids = kmeans.cluster_centers_
klabels = kmeans.labels_

from robotikos.fgm import fgn_shuffles
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

elbow_check = []
theta_range = [5.365]  # np.linspace(3.8, 5.5, num=40) # np.linspace(1, 3.5, num=50)
theta_range = [7.25] # 5.8 (104), 6.0 (101), 6.2 (102), 6.4 (101), 6.6 (102), 7.0 (101), 7.2 (101), 7.3 (99), 7.5 (98), 8.0 (96)
for t in theta_range:
    print(t)
    t1 = time()
    res = fgn_shuffles(points[:, None, :], thresholds=[t], metric=metric,
                       num_shuffles=100, mode='fgn_ma')
    t2 = time()

    fgn_time = t2 - t1
    print('fgn ', fgn_time)
    print('Speedup', kmeans_time/fgn_time)

    fcentroids = res[1]['centroids']
    flabels = res[1]['labels']
    print('# centroids', len(fcentroids.keys()))
    elbow_check.append(len(fcentroids.keys()))

fgn_centroids = np.zeros((len(fcentroids), 2))
for i, c in enumerate(fcentroids.keys()):
    fgn_centroids[i] = np.squeeze(fcentroids[c])


def plot_centroids_clusters(data, centroids, labels, title):

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

    pl.show()


plot_centroids_clusters(points, kcentroids, klabels, 'kmeans++')

plot_centroids_clusters(points, fgn_centroids, flabels, 'fgn_ma')

pl.plot(theta_range, elbow_check)
pl.grid()
pl.show()

t = 1

