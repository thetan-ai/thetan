import numpy as np
from time import time
from thetan import berserker, assign_pexact, centroids_from_labels
from thetan.viz import plot_centroids_clusters
import ray


@ray.remote
def process_chunk(data, start, end, theta_value):
    """
    Ray worker: process a slice of the dataset.
    Must copy slice because Ray object store gives read-only buffers.
    """
    # Ray gives a non-writeable buffer → must copy
    # X = np.array(data[start:end], copy=True)
    # X.setflags(write=True)
    X = data[start:end]

    cents, labels = berserker(X, theta=theta_value, metric="euclidean")
    return cents, labels


# @ray.remote
# def process_chunk(data, theta_value):
#     """
#     Ray worker: process a slice of the dataset.
#     """
#     X = np.array(data, copy=True)
#     # X = np.asarray(data)
#     X.setflags(write=True)

#     cents, labels = berserker(X, theta_value=theta_value, metric="euclidean")
#     return cents, labels


def berserker_papprox(points, theta_value, metric='euclidean'):
    """
    Parallel approximate berserker clustering using Ray.
    Returns per-chunk centroids (no global merge yet).
    """

    # ray.init(include_dashboard=False,
    #          logging_level="ERROR", log_to_driver=False)

    # Ensure contiguous, writeable array before putting into Ray
    points = np.ascontiguousarray(points, dtype=np.float64)
    # X.setflags(write=True)

    # # Store in Ray object store
    X_id = ray.put(points)

    # Compute chunk size
    n_cpu = int(ray.available_resources().get("CPU", 1))

    chunk_size = max(1, len(points) // n_cpu)

    futures = []
    for i in range(0, len(points), chunk_size):
        end = min(i + chunk_size, len(points))

        futures.append(process_chunk.remote(X_id, i, end, theta_value))

    # Retrieve all results
    results = ray.get(futures)   # list of (centroids_i, labels_i)

    # Combine all chunk centroids into one array
    all_centroids = [c for c, _ in results]

    # Optional: labels per chunk also returned
    all_labels = [lbl for _, lbl in results]

    # Carefully shutdown needs a few seconds
    # ray.shutdown()

    return all_centroids, all_labels






def make_grid_gaussians(A=4, B=3, l=5.0, sigma=1.0, n_per=150, d=2):
    rng = np.random.default_rng(424)
    xs = np.arange(-(A-1)/2, (A+1)/2) * l
    ys = np.arange(-(B-1)/2, (B+1)/2) * l
    C = np.array([(x, y) for x in xs for y in ys], dtype=float)
    X = np.vstack([c + rng.normal(scale=sigma, size=(n_per, d)) for c in C])
    y = np.repeat(np.arange(len(C)), n_per)
    return X, C, y




if __name__ == "__main__":

    t10 = time()
    ray.init(include_dashboard=False,
             logging_level="ERROR", log_to_driver=False)
    t11 = time()
    print("Ray init time: %.4f s" % (t11 - t10))

    X, Ctrue, _ = make_grid_gaussians(A=30, B=20, l=10.0, sigma=1.0, n_per=500)
    t12 = time()
    print("Data generation time: %.4f s" % (t12 - t11))

    np.random.shuffle(X)
    t13 = time()
    print("Data shuffling time: %.4f s" % (t13 - t12))

    print(X.shape)

    theta = 6

    t0 = time()
    centroids, labels = berserker(X, theta=theta, metric="euclidean")
    t1 = time()

    cents, labels = berserker_papprox(X, theta_value=theta, metric='euclidean')
    t2 = time()
    print("Berserker time: %.4f s" % (t1 - t0))
    print("Berserker papprox time: %.4f s" % (t2 - t1))


    print("Number of chunks:", len(cents))
    print("Done.")

    print("Berserker number of centroids:", len(centroids))
    print("Cents:", len(cents[0]))

    t = 1


    ray.shutdown()

    t3 = time()

    # Second phase

    all_cents = np.vstack(cents)
    t4 = time()

    final_cents, final_labels = berserker(all_cents, theta=t, metric='euclidean')
    t5 = time()
    print("After merging, number of centroids:", len(final_cents))

    labels = assign_pexact(X, final_cents, metric='euclidean')
    t6 = time()
    centroids = centroids_from_labels(X, labels)
    t7 = time()
    print("Recomputed centroids shape:", centroids.shape)

    print(f"Second phase times: shutdown ray {t3 - t2:.4f}, s gather {t4 - t3:.4f} s, merge {t5 - t4:.4f} s, assign {t6 - t5:.4f} s, recompute cents {t7 - t6:.4f} s")

    sum = t3 - t2 + t4 - t3 + t5 - t4 + t6 - t5 + t7 - t6
    print(f"Total second phase time: {sum:.4f} s")

    plot_centroids_clusters(X, centroids, labels, title="berserker papprox final clusters")

