import numpy as np
from sklearn.cluster import DBSCAN

from thetan.core import (
    candidate_labels_c,
    centroids_from_labels,
    reassign_uncertain_update_labels,
    sequential,
    sequential_prelim,
)


def berserker(points, theta, metric="euclidean"):
    """Cluster data with Thetan Berserker (TB).

    TB finds the number of clusters from a single distance threshold,
    theta. It runs Thetan Sequential (TS) twice on different random
    orderings of the data and cleans up spurious centroids after each pass,
    which makes the result much less sensitive to data order than a single
    TS pass.

    Parameters
    ----------
    points : ndarray, shape (n_samples, n_features)
        Data to cluster, as a C-contiguous float64 array.
    theta : float
        Distance threshold. A point joins its nearest centroid when their
        distance is below theta; otherwise it starts a new cluster. A good
        value is larger than the biggest cluster diameter and smaller than
        the smallest gap between clusters.
    metric : str, optional
        Distance metric, any name accepted by distance_matrix. The
        threshold theta is in the units of this metric. Default is
        'euclidean'. 'dot_prod' is the negative dot product, so theta is
        normally negative: theta=-0.9 means a point joins a centroid when
        their dot product is above 0.9. The dot product also grows with
        the length of the vectors, so rows are usually normalised to unit
        length first.

    Returns
    -------
    even_centroids_ex2 : ndarray, shape (n_clusters, n_features)
        Mean of the points in each cluster.
    updated_labels_ex2 : ndarray, shape (n_samples,)
        Cluster index of each point, in the original order of points.
    """
    ind = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
    points2 = points[ind].copy()
    ind_inv = np.argsort(ind)

    initial_centroids, initial_labels = sequential(
        points2, theta=theta, metric=metric, all_out=False
    )

    c_centroids, c_labels = sequential(
        initial_centroids, theta=theta, metric=metric, all_out=False
    )

    nice, candidates, _clusters, updated_labels = candidate_labels_c(
        c_centroids, c_labels, initial_labels
    )

    even_centroids = np.delete(initial_centroids, candidates, axis=0)

    del initial_centroids, initial_labels, candidates, _clusters, c_centroids, c_labels

    updated_labels2 = reassign_uncertain_update_labels(
        points2, even_centroids, updated_labels, nice, metric=metric
    )

    del nice, even_centroids, updated_labels

    even_centroids_ex = centroids_from_labels(points2, updated_labels2)

    del points2, updated_labels2

    ind2 = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
    points3 = points[ind2].copy()
    ind2_inv = np.argsort(ind2)

    p_centroids, p_labels = sequential_prelim(
        points3, even_centroids_ex, theta=theta, metric=metric, all_out=False
    )

    c_centroids2, c_labels2 = sequential(
        p_centroids, theta=theta, metric=metric, all_out=False
    )

    nice2, candidates2, _clusters2, updated_labels2x = candidate_labels_c(
        c_centroids2, c_labels2, p_labels
    )

    del c_centroids2, c_labels2, p_labels, _clusters2

    even_centroids2 = np.delete(p_centroids, candidates2, axis=0)

    del p_centroids, candidates2

    updated_labels_ex2 = reassign_uncertain_update_labels(
        points3, even_centroids2, updated_labels2x, nice2, metric=metric
    )

    del points3, even_centroids2, updated_labels2x, nice2

    updated_labels_ex2 = updated_labels_ex2[ind2_inv].copy()
    even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)

    return even_centroids_ex2, updated_labels_ex2


def tbscan(X, theta, eps=2.0, min_samples=1):
    """Cluster data with TBSCAN (Thetan Berserker followed by DBSCAN).

    TB first reduces the data to a small set of centroids. DBSCAN then
    groups those centroids, which lets TBSCAN find non-linearly separable
    clusters (rings, moons, spirals) at a fraction of the cost of running
    DBSCAN on every point.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data to cluster, as a C-contiguous float64 array.
    theta : float
        Distance threshold passed to berserker. TB always uses the
        Euclidean metric here.
    eps : float, optional
        DBSCAN neighbourhood radius between TB centroids, in the units of
        X. Default is 2.0.
    min_samples : int, optional
        Number of TB centroids (not points) a DBSCAN neighbourhood needs
        for its centre to be a core centroid. With the default of 1, every
        centroid is a core point and no point is labelled as noise.

    Returns
    -------
    centroids : ndarray, shape (n_clusters, n_features)
        Mean of the TB centroids in each DBSCAN cluster. This is an
        unweighted mean of centroids, not the mean of the points.
    labels : ndarray, shape (n_samples,)
        Cluster index of each point, in the original order of X. Points
        whose TB centroid DBSCAN marks as noise get -1; noise centroids
        are not included in centroids.
    """
    initial_centroids, initial_labels = berserker(X, theta)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(initial_centroids)

    unq = np.unique(model.labels_)
    db_nclusters = np.sum(unq >= 0)

    dbcentroids = np.array(
        [
            np.mean(initial_centroids[model.labels_ == l], axis=0)
            for l in range(db_nclusters)
        ]
    )
    dblabels = model.labels_

    final_labels = initial_labels.copy()

    for i, l in enumerate(initial_labels):
        final_labels[i] = dblabels[l]

    centroids = dbcentroids
    labels = final_labels

    return centroids, labels

