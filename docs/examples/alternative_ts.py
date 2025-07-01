import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from thetan.berserker import theta_g1 as ts_orig
from thetan.berserker import berserker2_clean as tb
from thetan.berserker import theta_means as tm

def theta_g1_python(X: np.ndarray, theta: float):
    N, D = X.shape
    centroids = []
    labels = np.full(N, -1, dtype=int)
    sums = []
    sizes = []

    for i in range(N):
        x = X[i]
        assigned = False
        for k, c in enumerate(centroids):
            dist = np.linalg.norm(x - c)
            if dist < theta:
                labels[i] = k
                sizes[k] += 1
                sums[k] += x
                centroids[k] = sums[k] / sizes[k]
                assigned = True
                break
        if not assigned:
            k = len(centroids)
            centroids.append(x.copy())
            sums.append(x.copy())
            sizes.append(1)
            labels[i] = k

    return np.vstack(centroids), np.array(labels)


def ts(X, theta):
    centroids = []
    counts = []

    for x in X:
        if not centroids:
            centroids.append(x.copy())
            counts.append(1)
            continue

        sims = np.sqrt(np.sum((np.array(centroids) - x) ** 2, axis=1))
        i_star = np.argmin(sims)
        min_sim = sims[i_star]

        if min_sim < theta:
            # Assign to best centroid
            counts[i_star] += 1
            eta = 1.0 / counts[i_star]
            centroids[i_star] = (1 - eta) * centroids[i_star] + eta * x
        else:
            # Create new centroid
            centroids.append(x.copy())
            counts.append(1)

    return np.array(centroids)


def ts_alt(X, theta):
    centroids = []
    counts = []

    for x in X:
        if not centroids:
            centroids.append(x.copy())
            counts.append(1)
            continue

        sims = np.sqrt(np.sum((np.array(centroids) - x) ** 2, axis=1))
        i_star = np.argmin(sims)
        min_sim = sims[i_star]

        if min_sim < theta:
            # Assign to best centroid
            counts[i_star] += 1
            eta = 1.0 / counts[i_star]
            centroids[i_star] = (1 - eta) * centroids[i_star] + eta * x
        else:
            # Create new centroid
            centroids.append(x.copy())
            counts.append(1)

    return np.array(centroids)

def ts_alt2(X, sigma, affinity_thresh):
    centroids = []
    counts = []

    for x in X:
        if not centroids:
            centroids.append(x.copy())
            counts.append(1)
            continue

        sims = np.exp(-np.sum((np.array(centroids) - x) ** 2, axis=1) / (2 * sigma**2))
        i_star = np.argmax(sims)
        max_sim = sims[i_star]

        if max_sim >= affinity_thresh:
            # Assign to best centroid
            counts[i_star] += 1
            eta = 1.0 / counts[i_star]
            centroids[i_star] = (1 - eta) * centroids[i_star] + eta * x
        else:
            # Create new centroid
            centroids.append(x.copy())
            counts.append(1)

    return np.array(centroids)


# --- Generate 2D disks with uniform samples ---
def sample_disk(center, radius, n):
    r = radius * np.sqrt(np.random.rand(n))
    t = 2 * np.pi * np.random.rand(n)
    x = center[0] + r * np.cos(t)
    y = center[1] + r * np.sin(t)
    return np.vstack((x, y)).T

def generate_disks(n_disks, spacing, radius, n_points):
    centers = [np.array([i * spacing, 0.0]) for i in range(n_disks)]
    samples = [sample_disk(center, radius, n_points) for center in centers]
    return np.vstack(samples), np.array(centers)

# --- Parameters ---
theta = 1.0
radius = theta / 2
n_points_per_disk = 10000
n_disks = 5
spacing = 0.8 * theta
sigma = radius / 2           # Controls decay of similarity

# affinity_thresh = 0.005         # Controls cluster granularity

# --- Generate dataset ---
X, true_centers = generate_disks(n_disks, spacing, radius, n_points_per_disk)

def randomize(X, ret_ind=False):
    ind = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=False)
    X2 = X[ind].copy()
    ind_inv = np.argsort(ind)
    if ret_ind:
        return X2, ind, ind_inv
    return X2

X = randomize(X)

# --- Run Gaussian TS ---
centroids, labels = ts_orig(X, theta)
print('Centroids')
print(centroids.shape)

centroids2 = ts(X, 0.7 * theta)
print('Centroids2')
print(centroids2.shape)

centroids3, labels3 = tb(X, theta_value=0.7 * theta)
print('Centroids3')
print(centroids3.shape)


centroids4 = ts(X, 0.05 * theta)
print('Centroids4')
print(centroids4.shape)

centroids4 = randomize(centroids4)

centroids5, labels5 = tb(centroids4, 0.7 * theta)
print('Centroids5')
print(centroids5.shape)

# we could now remove the overlapping parts.

# centroids6, labels6 = tm(X, centroids5, theta=0.7 * theta)
# print('Centroids6')
# print(centroids6.shape)

# --- Plot result ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.3, label="Data Points")
# ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.3, c=labels3, cmap='viridis', label="X")


# ax.scatter(centroids[:, 0], centroids[:, 1], color='red', s=40, label="Centroids")
# ax.scatter(centroids2[:, 0], centroids2[:, 1], color='blue', s=40, label="Centroids2")
# ax.scatter(centroids3[:, 0], centroids3[:, 1], color='orange', s=20, label="Centroids3")
# ax.scatter(centroids4[:, 0], centroids4[:, 1], color='purple', s=10, label="Centroids4")
# ax.scatter(centroids5[:, 0], centroids5[:, 1], color='green', s=40, label="Centroids5")   
# ax.scatter(centroids6[:, 0], centroids6[:, 1], color='orange', s=40, label="Centroids6")   


for c in true_centers:
    ax.add_patch(Circle(c, radius, fill=False, edgecolor='green', lw=1, linestyle='dashed'))
ax.set_title("TB on Multiple Merging Disks\nRed = Detected Centroids, Green = True Disk Centers")
ax.set_aspect("equal")
ax.grid(True)
# ax.legend()
plt.tight_layout()
plt.show()
