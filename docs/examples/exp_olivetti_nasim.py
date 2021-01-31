import logging
from time import time
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as pl

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from sklearn import decomposition
from robotikos.metric import IdentityFeature
from sklearn.cluster import KMeans
from robotikos.metricspeed import distance_matrix
from theta_machines_tests import theta_g1, theta_g2, theta_g2x, labels_to_centroids

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# #############################################################################
# Load faces data
faces, target = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                     random_state=rng)

# for f in range(faces.shape[0]):
#     face0 = faces[f].reshape((64, 64))
#     target0 = target[f]

#     plt.figure()
#     plt.title(target0)
#     plt.imshow(face0)
#     plt.show()
faces = faces.astype(np.float64)
face0 = faces[0].reshape((64, 64))
target0 = target[0]


n_samples, n_features = faces.shape
faces_centered = faces - faces.mean(axis=0)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
print("Dataset consists of %d faces" % n_samples)
test = 1

from theta_machines_tests import theta_g1, theta_g2, theta_g2x

# centroids, indices = theta_g1(faces, theta=1000)
methods = ['kmeans', 'kmeans++', 'fgn_ma', 'birch', 'mb_kmeans', 'gmm']
misses = {}
timings = {}
evaluation_thr = 0.5
if 'kmeans++' in methods:
    np.random.seed(0)
    t1 = time()
    kmeans = KMeans(n_clusters=40, init='k-means++', n_init=10,
                    max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(faces)
    t2 = time()
    kmeans_time = t2 - t1
    print('kmeans++ ', kmeans_time)
    kcentroids = kmeans.cluster_centers_
    klabels = kmeans.labels_
    timings['kmeans++'] = kmeans_time
    tmp = 0
test_fig = kcentroids.reshape(40,64,64)
pl.imshow(test_fig[1])
pl.show()
t = 1


num_clusters = []
range_theta = range(1, 65)
for theta_t in range_theta:
    centroids, indices = theta_g1(faces, theta=theta_t)
    num_clusters.append(len(centroids))

num_clusters = np.asarray(num_clusters)
pl.plot(range_theta, num_clusters)
pl.grid()
pl.show()


