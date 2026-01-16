import numpy as np
from thetan import berserker

def fit_tbknn(X, theta):
    """
    Build a TB-based KNN index from data X.

    Parameters:
    X : Input data of shape (n_samples, n_features)
    theta: the threshold value

    Returns:
    A dictionary containing (X, theta, labels, centroids, cluster_members)
    """

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D array of shape (n_samples, n_features).")

    n_samples, n_features = X.shape

    centroids, data_labels = berserker(X, theta)
    data_labels =  data_labels.ravel()

    if data_labels.shape[0] != n_samples:
        print(f'Data Labels:{data_labels.shape}')
        print(f'n_samples:{n_samples}')
        raise ValueError("Number of data labels does not match number of samples.")


    if centroids.ndim != 2 or centroids.shape[1] != n_features:
        raise ValueError(f"Centroids must have shape (K, {n_features}) and not {centroids.shape}")

    K = centroids.shape[0]
    cluster_members = [np.where(data_labels==k)[0] for k in range(K)]

    return {'X':X, 'theta':theta, 'labels':data_labels,'centroids':centroids,'cluster_members':cluster_members}

def nearest_clusters(centroids,q):
    """
    Computing the distances from query point to all centroids and return sorted order.
    """
    diff = centroids - q
    dists = np.linalg.norm(diff, axis=1)
    order = np.argsort(dists)
    return dists, order


def query_tbknn(index, q, k=10, nprobe=3, return_distances=True, reject_out_of_radius=False):
    """
    Queries the TBKNN index for appropriate nearest neighbors.

    Parameters:
    index: dictionary from fit_tbknn
    q: query point of shape (n_features,)
    k: Number of nearest neighbors to return
    nprobe: Number of nearest clusters to consider.
    return_distances: Bool for returning distances.
    reject_out_of_radius: Whether to reject points outside theta radius.

    Returns:
    indices: Indices of the k nearest neighbors.
    distances: Distances to the k nearest neighbors.
    """
    X = index['X']
    theta = index['theta']
    centroids = index['centroids']
    cluster_members = index['cluster_members']
    q = np.asarray(q, dtype=float).ravel()

    if q.ndim != 1 or q.shape[0] != X.shape[1]:
        raise ValueError(f"q must have shape ({X.shape[1]},).")

    if k<=0:
        raise ValueError("k must be a positive integer.")

    K = centroids.shape[0]
    nprobe = max(1, min(nprobe,K))

    centroid_dists, order = nearest_clusters(centroids, q)

    if reject_out_of_radius and centroid_dists[order[0]] > theta:
        if return_distances:
            return np.array([], dtype=int), np.array([],dtype=float)
        else:
            return np.array([],dtype=int)

    candidate_indices = []
    for c_idx in order[:nprobe]:
        members = cluster_members[c_idx]
        if members.size > 0:
            candidate_indices.append(members)

    if not candidate_indices:
        if return_distances:
            return np.array([], dtype=int), np.array([],dtype=float)
        else:
            return np.array([],dtype=int)

    candidate_indices = np.concatenate(candidate_indices, axis=0)
    candidate_points = X[candidate_indices]

    diff = candidate_points - q
    dists = np.linalg.norm(diff, axis = 1)

    order_cand = np.argsort(dists)[:k]
    neighbor_indices = candidate_indices[order_cand]
    neighbor_dists = dists[order_cand]

    if return_distances:
        return neighbor_indices, neighbor_dists
    else:
        return neighbor_indices

def tbknn(X, q, k=10, theta=0.5, nprobe=3, index=None):
    """
    Finding the Approximate k nearest neighbors.

    X : Input data of shape (n_samples, n_features)
    q : query point of shape (n_features,)
    k : Number of nearest neighbors to return.
    theta: the threshold value.
    nprobe: Number of nearest clusters to probe.
    index: The fitted index for reusing for multiple queries on the same data.
    """

    if index is None:
        index = fit_tbknn(X, theta)

    indices, distances = query_tbknn(index, q, k, nprobe)

    return indices, distances, index
