import numpy as np
import matplotlib.pyplot as plt
from thetan import berserker
from thetan.metrics import distance_matrix
from thetan.viz import visualize_separation
from PIL import Image



def one_hop_counts(D, eps, directed=False, return_adjacency=False):
    """
    Counts the 1-hop connections per point from an (N X N) distance matrix.

    Parameters:
    * D : Distance matrix of shape (N x N).
    * eps : Radius for 1-hop connections.
    * directed : If True, count outgoing neighbors only. If False, symmetrize the adjacency matrix.
    * return_adjacency : Whether to return the adjacency matrix.

    Returns:
    * counts: number of 1-hop connections per point.
    * adjacency: adjacency matrix if return_adjacency is True.
    """

    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f'D must be a square matrix of shape N x N, but got shape {D.shape}')
    
    if not np.isfinite(eps) or eps < 0:
        raise ValueError(f'eps must be a non-negative real number, but got {eps}')
    
    A = D <= eps

    np.fill_diagonal(A, False)

    if not directed:
        A = A | A.T
    
    counts = A.sum(axis=1).astype(int)

    if return_adjacency:
        return counts, A
    
    return counts


def sparse_centroids(centroids, eps, min_neighbors=3, metric='euclidean'):
    """
    Finds the centroids which have fewer than min_neighbors.

    Parameters:
    * centroids: Cluster Centroids of shape (n_centroids, n_dims).
    * eps: Radius for 1-hop connections.
    * min_neighbors: Minimum number of neighbors required for a centroid to be considered non-sparse.
    * metric: Metric to use for distance calculation.

    Returns:
    * sparse_indices: Indices of the sparse centroids.
    * dense_indices: Indices of the dense centroids.
    * counts: Number of 1-hop connections per centroid.
    """

    dm = distance_matrix(centroids, centroids, metric='euclidean')
    counts = one_hop_counts(dm, eps=eps)
    sparse_mask = counts < min_neighbors
    sparse_indices = np.where(sparse_mask)[0]
    dense_indices = np.where(~sparse_mask)[0]
    
    return sparse_indices, dense_indices, counts

def remove_sparse_centroids(data, centroids, labels, sparse_indices):
    """
    Remove the data points belonging to the sparse centroids.

    Parameters:
    * data: Data points of shape (n_samples, n_dims).
    * centroids: Cluster Centroids of shape (n_centroids, n_dims).
    * labels: Cluster labels of shape (n_samples,).
    * sparse_indices: Indices of the sparse centroids.

    Returns:
    * filtered_data: Data points of shape (n_samples - n_sparse, n_dims).
    * filtered_centroids: Cluster Centroids of shape (n_centroids - n_sparse, n_dims).
    * filtered_labels: Cluster labels of shape (n_samples - n_sparse,).
    * removed_mask: Boolean mask of shape (n_samples,) indicating which data points were removed.
    """

    if len(sparse_indices) == 0:
        return data.copy(), centroids.copy(), labels.copy(), np.zeros(len(data), dtype=bool)
    
    removed_mask = np.isin(labels, sparse_indices)

    filtered_data = data[~removed_mask]
    all_indices = np.arange(len(centroids))
    dense_indices = np.setdiff1d(all_indices, sparse_indices)

    filtered_centroids = centroids[dense_indices]

    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(dense_indices)}
    old_labels = labels[~removed_mask]
    filtered_labels = np.array([label_map[label] for label in old_labels])

    return filtered_data, filtered_centroids, filtered_labels, removed_mask

def separate_density_regions(data, centroids, labels, eps, min_neighbors=3, metric='euclidean', verbose=True):
    """
    Separate connected density regions by removing sparse bridge centroids.

    Parameters:
    * data: Data points of shape (n_samples, n_dims).
    * centroids: Cluster Centroids of shape (n_centroids, n_dims).
    * labels: Cluster labels of shape (n_samples,).
    * eps: Radius for counting neighbors between centroids.
    * min_neighbors: Minimum number of neighbors required for a centroid to be considered non-sparse.
    * metric: Metric to use for distance calculation.
    * verbose: Whether to print progress.

    Returns:
    * result: dict with keys:
        * filtered_data : Data after removing bridge points.
        * filtered_centroids: Centroids after removing sparse ones.
        * filtered_labels: Reindexed labels.
        * removed_mask: Boolean mask of shape (n_samples,) indicating which data points were removed.
        * sparse_indices: Indices of the sparse centroids.
        * dense_indices: Indices of the dense centroids.
        * counts: Neighbor counts for original centorids.
        * n_dims: Number of dimensions in the data.
    """

    n_points, n_dims = data.shape
    n_centroids = len(centroids)

    if verbose:
        print(f'Data shape: {data.shape} ({n_dims} dimensions)')
        print(f'Number of centroids: {n_centroids}')
        print(f'Neighbor eps: {eps}')
        print(f'Minimum neighbors threshold: {min_neighbors}')
        print(f'Distance metric: {metric}')
    
    sparse_indices, dense_indices, counts = sparse_centroids(centroids, eps, min_neighbors, metric)

    if verbose:
        print(f'Number of sparse centroids: {len(sparse_indices)}')
        if len(sparse_indices) > 0:
            print(f'Sparse indices: {sparse_indices}')
            print(f'Their neighbors: {counts[sparse_indices]}')
    
    filtered_data, filtered_centroids, filtered_labels, removed_mask = remove_sparse_centroids(data, centroids, labels, sparse_indices)

    if verbose:
        print(f'Points removed: {np.sum(removed_mask)}/ {n_points}')
        print(f'Centroids Remaining: {len(filtered_centroids)} / {n_centroids}')
    
    return {
        'filtered_data' : filtered_data,
        'filtered_centroids' : filtered_centroids,
        'filtered_labels' : filtered_labels,
        'removed_mask' : removed_mask,
        'removed_data' : data[removed_mask],
        'sparse_indices' : sparse_indices,
        'dense_indices' : dense_indices,
        'counts' : counts,
        'n_dims' : n_dims
    }

def load_image_points(image_path, threshold=50, max_points=20000, seed=None):
    """
    Load 2D points from a binary image.

    Parameters:
    * image_path : Path to image file.
    * threshold : Threshold for binarization.
    * max_points : Maximum number of points to sample.
    * seed : Random seed for reproducibility.

    Returns:
    * coords : Array of points.
    """

    img = Image.open(image_path).convert("L")
    pixels = np.array(img)
    binary = pixels > threshold

    coords = np.column_stack(np.where(binary.T))

    n_points = min(max_points, len(coords))
    if n_points < len(coords):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(coords), size=n_points, replace=False)
        coords = coords[indices]

    return np.ascontiguousarray(coords, dtype=np.float64)

def sample_ball(center, radius, n_points, n_dim, rng=None):
    """
    Sample points uniformly from an n-dimensional ball.
    """
    if rng is None:
        rng = np.random.default_rng()
    directions = rng.standard_normal((n_points, n_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    radii = radius * (rng.random(n_points) ** (1.0/n_dim))
    return center + directions * radii[:, np.newaxis]

def generate_nd_connected_balls(n_dim=3, n_points_per_ball=5000, n_bridge_points=500, ball_radius=50.0, ball_separation=150.0, bridge_width=10.0, seed=None):
    """
    Generate two synthetic n-dimensional data balls connected by a bridge. 

    Parameters:
    * n_dims: Number of dimensions.
    * n_points_per_ball: Points in each ball.
    * n_bridge_points: Points in the bridge,
    * ball_radius: Radius of the balls.
    * ball_separation: Separation distance between the balls.
    * bridge_width: Width of the bridge.
    * seed: Random seed for reproducibility.

    Returns:
    * data: Array of shape (n_total_points, n_dim).
    * ground_truth: Array of shape (n_total_points,).
    0 = left ball, 1 = right ball, 2 = bridge.
    """

    rng = np.random.default_rng(seed)
    
    ball1_center = np.zeros(n_dim)
    ball2_center = np.zeros(n_dim)
    ball2_center[0] = ball_separation
    
    ball1 = sample_ball(ball1_center, ball_radius, n_points_per_ball, n_dim, rng=rng)
    ball2 = sample_ball(ball2_center, ball_radius, n_points_per_ball, n_dim, rng=rng)

    bridge_start = ball_radius * 0.8
    bridge_end = ball_separation - ball_radius * 0.8

    bridge = np.zeros((n_bridge_points, n_dim))
    bridge[:, 0] = rng.uniform(bridge_start, bridge_end, n_bridge_points)

    if n_dim > 1:
        noise = rng.standard_normal((n_bridge_points, n_dim - 1)) * bridge_width
        bridge[:, 1:] = noise
    
    data = np.vstack([ball1,ball2,bridge])

    ground_truth = np.concatenate([
        np.zeros(n_points_per_ball),
        np.ones(n_points_per_ball),
        np.full(n_bridge_points,2)
    ]).astype(int)

    shuffle_idx = rng.permutation(len(data))
    data = data[shuffle_idx]
    ground_truth = ground_truth[shuffle_idx]

    return data.astype(np.float64), ground_truth

def pipeline(data, theta, eps, min_neighbors=3, metric='euclidean', visualize=True, save_prefix=None, verbose=True):
    """
    Pipeline to separate connected density regions.

    Parameters:
    * data: Input data of shape (n_points, n_dims).
    * theta : threshold for clustering.
    * eps: Radius for counting centroid neighbors.
    * min_neighbors: Minimum number of neighbors to be considered dense. Anything less would be considered sparse.
    * metric: Distance metric.
    * visualize: If True, visualize the results.
    * save_prefix: Prefix for saving the results.
    * verbose: If True, print progress.

    Returns:
    * result: Dictionary containing all the filtering results, original_centroids, and original_labels.
    """

    n_points, n_dims = data.shape

    if verbose:
        print('Clustering using Thetan Berserker...')
    
    centroids, labels = berserker(data, theta=theta)

    if verbose:
        print(f'Identifying and removing sparse centroids...')
    
    result = separate_density_regions(data, centroids, labels, eps=eps, min_neighbors=min_neighbors, metric=metric, verbose=verbose)

    result['original_centroids'] = centroids
    result['original_labels'] = labels
    result['original_data'] = data

    if visualize:
        if verbose:
            print('Generating visualizations...')

        save_path = f'{save_prefix}_separation.png' if save_prefix else None
        if verbose:
            print(f'Saving visualizations as {save_path}')
        visualize_separation(data, centroids, labels, result, save_path=save_path, show=True)

    return result

if __name__ == '__main__':
    # For 2d image
    data = load_image_points('Density_balls.png',threshold=50,max_points=50000)
    theta = 50
    eps = 85 
    result = pipeline(data, theta=theta, eps=eps, min_neighbors=3,visualize=True,save_prefix='2dimage',verbose=True)

    # For 3D Example
    # data, ground_truth = generate_nd_connected_balls(n_dim=3, n_points_per_ball=3000, n_bridge_points=300, ball_radius=50,ball_separation=150,bridge_width=8,seed=42)
    # print(f'Generated {len(data)} points')
    # print(f"Ground truth: {np.sum(ground_truth == 0)} ball1, {np.sum(ground_truth == 1)} ball2, {np.sum(ground_truth == 2)} bridge")
    # result = pipeline(data, theta=30, eps=90, min_neighbors=3, visualize=True, save_prefix='3d_synthetic', verbose=True)
