import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from thetan.berserker import theta_g1 as ts

# Parameters
theta = 0.8
r = 0.5
ell = 0.6  # center separation
K = 15
n_per_ball = 10000
dimensions = [100, 200]
# dimensions = [2, 3, 4, 5, 10, 20, 50, 100, 200]
num_trials = 2 # 100
uplift = True
# np.random.seed(42)

# Generate K centers aligned along X-axis
def generate_axis_aligned_centers(d, K, ell):
    centers = np.zeros((K, d))
    for i in range(K):
        centers[i, 0] = i * ell
    return centers


def generate_aligned_centers(d, K, ell, axis=None, seed=None):
    """
    Generate K cluster centers in d-dimensional space, aligned along a given or random axis.

    Parameters:
    - d (int): Dimension of the space.
    - K (int): Number of centers to generate.
    - ell (float): Distance between adjacent centers.
    - axis (ndarray or None): Optional unit vector of shape (d,) defining alignment direction.
                               If None, a random unit vector is generated.
    - seed (int or None): Optional random seed for reproducibility.

    Returns:
    - centers (ndarray): Array of shape (K, d), aligned along the specified axis.
    - axis (ndarray): The unit direction vector used.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate or normalize the alignment axis
    if axis is None:
        axis = np.random.randn(d)
    axis = axis / np.linalg.norm(axis)

    # Generate centers along the axis
    scalars = np.arange(K) * ell
    centers = np.outer(scalars, axis)  # shape: (K, d)

    return centers, axis



# Generate uniform samples inside a d-ball
def generate_uniform_ball(center, radius, n, d):
    directions = np.random.normal(size=(n, d))
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    radii = np.random.rand(n) ** (1 / d)
    return center + radius * (radii[:, np.newaxis] * directions)

# Thetan Sequential (TS) clustering using neighbor graph + connected components
def approx_thetan_sequential_as_cc(X, theta):
    nbrs = NearestNeighbors(radius=theta, metric='euclidean').fit(X)
    graph = nbrs.radius_neighbors_graph(X, mode='connectivity')
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return labels

def randomize(X, ret_ind=False):
    ind = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=False)
    X2 = X[ind].copy()
    ind_inv = np.argsort(ind)
    if ret_ind:
        return X2, ind, ind_inv
    return X2

# Run the experiment
results = {d: [] for d in dimensions}

for d in dimensions:
    for trial in range(num_trials):
        # centers = generate_axis_aligned_centers(d, K, ell)
        centers, axis = generate_aligned_centers(d, K, ell)
        X = np.vstack([generate_uniform_ball(center, r, n_per_ball, d) for center in centers])
        X = randomize(X)
        if uplift:
            centroids, labels = ts(X, 0.05)
            centroids, labels = ts(centroids, theta)
        else:    
            centroids, labels = ts(X, theta)
        num_clusters = len(set(labels))
        results[d].append(num_clusters)
        print(f"Trial {trial + 1}/{num_trials} for dimension {d}, found {num_clusters} clusters.")
        

# Plot results
plt.figure(figsize=(10, 6))
means = [np.mean(results[d]) for d in dimensions]
stds = [np.std(results[d]) for d in dimensions]

plt.errorbar(dimensions, means, yerr=stds, fmt='o-', capsize=5)
plt.axhline(K, color='gray', linestyle='--', label=f'True number of clusters (K={K})')
plt.xlabel(f"Dimension {d}")
plt.ylabel("Mean number of clusters detected by TS")
plt.title(f"TS Separation Accuracy vs Dimension (Axis-Aligned, $\\ell$={ell}, $\\theta$={theta}, r={r}, K={K}, Real TS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
