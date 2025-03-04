import numpy as np

def cross_entropy(p, q):
    """
    Compute categorical cross-entropy between two distributions.
    """
    epsilon = 1e-10  # Avoid log(0)
    q = np.clip(q, epsilon, 1.0)
    return -np.sum(p * np.log(q))

def initialize_clusters(data, k):
    """
    Randomly initialize cluster prototypes.
    """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, clusters):
    """
    Assign each data point to the cluster with the smallest cross-entropy distance.
    """
    assignments = []
    for p in data:
        distances = [cross_entropy(p, q) for q in clusters]
        assignments.append(np.argmin(distances))
    return np.array(assignments)

def update_clusters(data, assignments, k):
    """
    Update cluster prototypes as the mean of assigned points.
    """
    new_clusters = []
    for i in range(k):
        points_in_cluster = data[assignments == i]
        if len(points_in_cluster) > 0:
            new_cluster = np.mean(points_in_cluster, axis=0)
        else:  # Handle empty clusters
            new_cluster = np.random.dirichlet(np.ones(data.shape[1]))
        new_clusters.append(new_cluster)
    return np.array(new_clusters)

def cross_entropy_clustering(data, k, max_iter=100):
    """
    Perform clustering using cross-entropy as a distance metric.
    """
    clusters = initialize_clusters(data, k)
    for _ in range(max_iter):
        assignments = assign_clusters(data, clusters)
        new_clusters = update_clusters(data, assignments, k)
        if np.allclose(clusters, new_clusters):
            break
        clusters = new_clusters
    return assignments, clusters

# Example: Clustering data points represented as probability distributions
data = np.random.dirichlet(np.ones(3), size=100)  # 100 data points, 3 classes
k = 3  # Number of clusters
assignments, clusters = cross_entropy_clustering(data, k)

print("Cluster Assignments:", assignments)
print("Cluster Centroids:", clusters)



t = 1