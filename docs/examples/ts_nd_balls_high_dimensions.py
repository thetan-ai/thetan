import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from thetan.berserker import theta_g1 as ts

# Parameters
theta = 1.0
r = theta / 2
ell = theta  # center separation
K = 15
n_per_ball = 10000
dimensions = [2, 3, 4, 5, 10, 20, 50, 100, 200]
num_trials = 100
# np.random.seed(42)

# Generate K centers aligned along X-axis
def generate_axis_aligned_centers(d, K, ell):
    centers = np.zeros((K, d))
    for i in range(K):
        centers[i, 0] = i * ell
    return centers

# Generate uniform samples inside a d-ball
def generate_uniform_ball(center, radius, n, d):
    directions = np.random.normal(size=(n, d))
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    radii = np.random.rand(n) ** (1 / d)
    return center + radius * (radii[:, np.newaxis] * directions)

# Thetan Sequential (TS) clustering using neighbor graph + connected components
def thetan_sequential_as_cc(X, theta):
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
        print(f"Trial {trial + 1}/{num_trials} for dimension {d}")
        centers = generate_axis_aligned_centers(d, K, ell)
        X = np.vstack([generate_uniform_ball(center, r, n_per_ball, d) for center in centers])
        X = randomize(X)
        centroids, labels = ts(X, theta)
        num_clusters = len(set(labels))
        results[d].append(num_clusters)

# Plot results
plt.figure(figsize=(10, 6))
means = [np.mean(results[d]) for d in dimensions]
stds = [np.std(results[d]) for d in dimensions]

plt.errorbar(dimensions, means, yerr=stds, fmt='o-', capsize=5)
plt.axhline(K, color='gray', linestyle='--', label=f'True number of clusters (K={K})')
plt.xlabel("Dimension $d$")
plt.ylabel("Mean number of clusters detected by TS")
plt.title("TS Separation Accuracy vs Dimension (Axis-Aligned, $\\ell$, K=15, Real TS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
