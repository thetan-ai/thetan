from time import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from thetan.berserker import berserker2_clean
from thetan.viz import plot_centroids_clusters


def dbscan(X, eps=0.5, min_samples=1):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)

    unq = np.unique(model.labels_)
    db_nclusters = np.sum(unq>=0)

    dbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(db_nclusters)])
    dblabels = model.labels_

    return dbcentroids, dblabels

def tbscan(X, theta_value, eps=0.5, min_samples=1):

    initial_centroids, initial_labels = berserker2_clean(X, theta_value)

    t = time()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(initial_centroids)

    unq = np.unique(model.labels_)
    db_nclusters = np.sum(unq>=0)

    dbcentroids = np.array([np.mean(initial_centroids[model.labels_==l], axis=0) for l in range(db_nclusters)])
    dblabels = model.labels_

    final_labels = initial_labels.copy()

    for i, l in enumerate(initial_labels):
        final_labels[i] = dblabels[l]

    duration = time() - t

    return dbcentroids, final_labels


# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 100000 # 150000
seed = 30
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# # ============
# # Set up cluster parameters
# # ============
# plt.figure(figsize=(9 * 2 + 3, 13))
# plt.subplots_adjust(
#     left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
# )

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {}),
]

dpath = 'C:/Users/Eleftherios/Data/tb_circles/'


for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    # params = default_base.copy()
    # params.update(algo_params)

    X, y = dataset

    # # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    print('Dataset size is ')
    print(X.shape)
    print(X.min(), X.max())

    t0 = time()
    centroids, labels = berserker2_clean(X, 0.5)
    t1 = time()

    plot_centroids_clusters(X, None, labels, 'TB ', show=False, savefn=dpath+'tb.png')

    ch = 1

    t2 = time()
    centroids, labels = tbscan(X, 0.5, eps=.8, min_samples=1)
    t3 = time()

    plot_centroids_clusters(X, None, labels, 'TBSCAN ', show=False, savefn=dpath+'tbscan.png')

    t4 = time()
    centroids, labels = dbscan(X, eps=0.1, min_samples=7)
    t5 = time()

    plot_centroids_clusters(X, None, labels, 'DBSCAN', show=False, savefn=dpath+'dbscan.png')

    plot_num +=1

    print('TB', t1-t0, 'TBSCAN', t3 - t2, 'DBSCAN', t5 - t4)


    print('Speedup ', (t5 - t4) / (t3 - t2))

    ch = 12
    break

