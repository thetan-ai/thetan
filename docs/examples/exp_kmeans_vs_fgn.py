from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)



import pylab as pl


sz = (50, 30)

coords = []
centroids = []

for x, y in np.ndindex(sz):

    for d in range(100):
        r = np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        x1 = 5 * x + r * np.cos(theta)
        y1 = 5 * y + r * np.sin(theta)
        coords.append([x1, y1])
    centroids.append([x, y])

points = np.array(coords)

pl.plot(points[:, 0], points[:, 1], '.')
pl.show()


from sklearn.cluster import KMeans

t1 = time()
kmeans = KMeans(n_clusters=np.prod(sz), random_state=0).fit(points)
t2 = time()

print('kmeans++ ', t2 - t1)

kcentroids = kmeans.cluster_centers_

from robotikos.fgm import fgn_shuffles
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

t1 = time()
res = fgn_shuffles(points[:, None, :], thresholds=[1.], metric=metric, num_shuffles=1, mode='fgn_ma')
t2 = time()

print('fgn ', t2 - t1)

fcentroids = res[1]['centroids']

fgn_centroids = np.zeros((len(fcentroids), 2))
for i, c in enumerate(fcentroids.keys()):
    fgn_centroids[i] = np.squeeze(fcentroids[c])

pl.figure()
pl.title('kmeans ++')
pl.plot(points[:, 0], points[:, 1], '.')
pl.plot(kcentroids[:, 0], kcentroids[:, 1], 'rx')

pl.figure()
pl.title('fgn')
pl.plot(points[:, 0], points[:, 1], '.')
pl.plot(fgn_centroids[:, 0], fgn_centroids[:, 1], 'rx')

pl.show()

t = 1

