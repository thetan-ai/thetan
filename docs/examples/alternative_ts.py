import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from thetan.berserker import theta_g1 as ts_orig
from thetan.berserker import berserker2_clean as tb
from thetan.berserker import theta_means as tm
from thetan.berserker import distance_matrix as dm


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

def plot_overlapping_disks(X, centroids_ph1, radius, mode=['init', 'phase2'], centroids_ph2=None, true_centers=None):

    # --- Plot result ---
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'init' in mode:  
        ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.3, label="Data Points")

    ax.scatter(centroids_ph1[:, 0], centroids_ph1[:, 1], color='purple', s=10, label="Centroids")
    
    if 'phase2' in mode:
        ax.scatter(centroids_ph2[:, 0], centroids_ph2[:, 1], color='green', s=40, label="Centroids2")   
    
    for c in true_centers:
        ax.add_patch(Circle(c, radius, fill=False, edgecolor='green', lw=1, linestyle='dashed'))

    ax.scatter(true_centers[:, 0], true_centers[:, 1], color='black', s=20, label="True Centers")

    ax.set_title("TB on Multiple Merging Disks\nRed = Detected Centroids, Green = True Disk Centers")
    ax.set_aspect("equal")
    ax.grid(True)
    # ax.legend()
    plt.tight_layout()
    plt.show()




# --- Parameters ---
theta = 1.0
radius = theta / 2
n_points_per_disk = 10000
n_disks = 15
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
print(f"Dataset shape: {X.shape}")

def tb_phase2(X, theta_little, theta_big):

    # centroids, labels = ts_orig(X, theta)
    # print('Centroids')
    # print(centroids.shape)

    # centroids2 = ts(X, 0.7 * theta)
    # print('Centroids2')
    # print(centroids2.shape)

    # centroids3, labels3 = tb(X, theta_value=0.7 * theta)
    # print('Centroids3')
    # print(centroids3.shape)

    centroids4 = ts(X, theta_little)
    # print('Centroids Little')
    # print(centroids4.shape)

    centroids4 = randomize(centroids4)

    centroids5, labels5 = tb(centroids4, theta_big)
    # print('Centroids Big')
    # print(centroids5.shape)

    return centroids4, centroids5


check_hyperparams = False

if check_hyperparams:

    for shrink in np.arange(0.65, 0.75, 0.01):
        for i in range(20):

            centroids4, centroids5 = tb_phase2(X, theta * 0.05, theta * shrink)
            
            distance_matrix = dm(centroids5, true_centers)
            min_distances_sum = np.min(distance_matrix, axis=1).sum()
            print(f"Shrink: {shrink:.2f}, Iteration: {i+1}, Min Distances Sum: {min_distances_sum:.2f}, Shape: {centroids5.shape[0]}" )
        print()

# shrink = 0.71
shrink = 0.64
centroids4, centroids5 = tb_phase2(X, theta * 0.05, theta * shrink)
dm_5t = dm(centroids5, true_centers)
min_distances_sum0 = np.min(dm_5t, axis=0).sum()
min_distances_sum1 = np.min(dm_5t, axis=1).sum()

print(f"Shrink: {shrink:.2f}, Min Sum0: {min_distances_sum0:.2f}, Min Sum1: {min_distances_sum1:.2f}, Shape: {centroids5.shape[0]}")

# shrink = 0.72
# centroids6, labels6 = tm(X, centroids5, theta=theta*shrink)
# dm_6t = dm(centroids6, true_centers)
# min_distances_sum0 = np.min(dm_6t, axis=0).sum()
# min_distances_sum1 = np.min(dm_6t, axis=1).sum()

# print(f"Shrink: {shrink:.2f}, Min Sum0: {min_distances_sum0:.2f}, Min Sum1: {min_distances_sum1:.2f}, Shape: {centroids6.shape[0]}" )

# we could now remove the overlapping parts.

# centroids6, labels6 = tm(X, centroids5, theta=0.7 * theta)
# print('Centroids6')
# print(centroids6.shape)




plot_overlapping_disks(X, centroids4, radius, mode=['phase2'],
                       centroids_ph2=centroids5, true_centers=true_centers)

# plot_overlapping_disks(X, centroids4, radius, mode=['phase2'],
#                        centroids_ph2=centroids6, true_centers=true_centers)
