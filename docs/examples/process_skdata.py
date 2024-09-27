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



def tbscan(X, theta_value, eps=0.5, min_samples=1):

    initial_centroids, initial_labels = berserker2_clean(X, theta_value)

    t = time()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(initial_centroids)

    unq = np.unique(model.labels_)
    db_nclusters = np.sum(unq>=0)

    dbcentroids = np.array([np.mean(initial_centroids[model.labels_==l], axis=0) for l in range(db_nclusters)])
    dblabels = model.labels_
    duration = time() - t


    final_labels = initial_labels.copy()

    for i, l in enumerate(initial_labels):
        final_labels[i] = dblabels[l]

    return final_labels


# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
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

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

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

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    for i in range(1):

        t0 = time()
        # centroids, labels = berserker2_clean(X, 0.1)
        labels = tbscan(X, 0.1, eps=.4, min_samples=1)
        t1 = time()

        y_pred = labels

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )


        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred, cmap=plt.cm.nipy_spectral)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        # plt.xticks(())
        # plt.yticks(())
        # plt.text(
        #     0.99,
        #     0.01,
        #     ("%.2fs" % (t1 - t0)).lstrip("0"),
        #     transform=plt.gca().transAxes,
        #     size=15,
        #     horizontalalignment="right",
        # )
        plot_num += 1

        plt.show()

