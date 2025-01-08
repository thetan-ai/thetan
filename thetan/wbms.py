import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def K(x, y, w, h):
    """
    Kernel function that computes a similarity value between two points.

    Args:
        x: First data point (numpy array).
        y: Second data point (numpy array).
        w: Feature weights (numpy array).
        h: Kernel bandwidth (float).

    Returns:
        float: Kernel similarity value.
    """
    return np.exp(-np.sum((x - y) ** 2 * w) / h)

def WBMS(X, h, lam=1, tmax=30):
    """
    Weighted Blurring Mean Shift clustering algorithm.

    Args:
        X: Data matrix (n_samples, n_features).
        h: Kernel bandwidth.
        lam: Regularization parameter for weights.
        tmax: Maximum number of iterations.

    Returns:
        tuple: Transformed dataset and feature weights.
    """
    n, p = X.shape
    K_matrix = np.zeros((n, n))
    w = np.ones(p) / p
    D = np.zeros(p)
    X1 = X.copy()
    X2 = X.copy()

    # Initialization phase (20 iterations)
    for t in range(20):
        for i in range(n):
            for j in range(i, n):
                K_matrix[i, j] = K(X2[i], X2[j], w, h)
                K_matrix[j, i] = K_matrix[i, j]  # Symmetry

        for i in range(n):
            I = np.delete(np.arange(n), i)
            s = np.sum(K_matrix[I, i])
            for l in range(p):
                X1[i, l] = np.sum(X2[I, l] * K_matrix[I, i])
            X1[i] /= s

        D = np.sum((X - X1) ** 2, axis=0)
        w = np.exp(-D / lam)
        w /= np.sum(w)
        X2 = X1.copy()
        print(f"Initialization iteration {t+1}")

    # Main phase
    for t in range(tmax):
        for i in range(n):
            for j in range(i, n):
                K_matrix[i, j] = K(X2[i], X2[j], w, h)
                K_matrix[j, i] = K_matrix[i, j]  # Symmetry

        for i in range(n):
            I = np.delete(np.arange(n), i)
            s = np.sum(K_matrix[I, i])
            for l in range(p):
                X1[i, l] = np.sum(X2[I, l] * K_matrix[I, i])
            X1[i] /= s

        D = np.sum((X - X1) ** 2, axis=0)
        w = np.exp(-D / lam)
        w /= np.sum(w)
        X2 = X1.copy()

        if t % 50 == 0:
            print(f"Main phase iteration {t+1}")

    return X2, w

def U2clus(U, epsa=1e-5):
    """
    Converts a transformed dataset into cluster labels.

    Args:
        U: Transformed dataset (n_samples, n_features).
        epsa: Distance threshold for adjacency.

    Returns:
        numpy array: Cluster membership for each point.
    """
    n = U.shape[0]
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if np.sqrt(np.sum((U[i] - U[j]) ** 2)) < epsa:
                A[i, j] = 1

    graph = csr_matrix(A)
    _, labels = connected_components(csgraph=graph, directed=False)

    return labels

# # Example usage
# if __name__ == "__main__":
#     # Generate a toy dataset
#     X1 = np.random.normal(size=(100, 2))
#     X2 = np.random.normal(loc=5, size=(100, 2))
#     X = np.vstack((X1, X2))
#     for _ in range(30):
#         X = np.hstack((X, np.random.normal(size=(200, 1))))

#     # Normalize the dataset
#     X = (X - X.mean(axis=0)) / X.std(axis=0)

#     # Plot the original data
#     import matplotlib.pyplot as plt
#     plt.scatter(X[:, 0], X[:, 1])
#     plt.title("Original Data")
#     plt.show()

#     # Run WBMS
#     h = 0.1
#     lam = 10
#     X2, w = WBMS(X, h, lam, tmax=30)

#     # Plot the centroids
#     plt.scatter(X2[:, 0], X2[:, 1], color='red', marker='o')
#     plt.title("Centroids")
#     plt.show()

#     # Print feature weights
#     print("Feature Weights:", w)

#     # Find clusters
#     labels = U2clus(X2)

#     # Plot clustered data
#     plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
#     plt.title("Clustered Data")
#     plt.colorbar(label="Cluster")
#     plt.show()




# Functions (Assumes WBMS and U2clus are already defined from previous implementation)
def generate_toy_data():
    """
    Generate a toy dataset with relevant and irrelevant features.
    Returns:
        X: The generated dataset.
    """
    X1 = np.random.normal(0, 1, (100, 2))
    X2 = np.random.normal(5, 1, (100, 2))
    X = np.vstack((X1, X2))

    for _ in range(30):
        irrelevant_feature = np.random.normal(0, 1, 200).reshape(-1, 1)
        X = np.hstack((X, irrelevant_feature))

    # Normalize the data
    X = zscore(X, axis=0)
    return X

# Generate toy data
X = generate_toy_data()

# Plot the original data
plt.scatter(X[:, 0], X[:, 1], label="Original Data", alpha=0.7)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Run the WBMS function
h = 0.1
lam = 10
tmax = 30
transformed_data, feature_weights = WBMS(X, h, lam, tmax)

# Plot the centroids
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color='red', label="Centroids", marker='o')
plt.title("Transformed Data with Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Print the feature weights
print("Feature Weights:")
print(feature_weights)

# Find the clusters
cluster_labels = U2clus(transformed_data)

# Plot data color-coded by cluster labels
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", label="Clustered Data")
plt.title("Clustered Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.show()
