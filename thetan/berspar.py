import numpy as np
from time import time
from thetan import berserker
import ray


ray.init()

@ray.remote
def process_chunk(data, start, end, theta_value):
    """
    Ray worker: process a slice of the dataset.
    Must copy slice because Ray object store gives read-only buffers.
    """
    # Ray gives a non-writeable buffer → must copy
    X = np.array(data[start:end], copy=True)
    X.setflags(write=True)

    cents, labels = berserker(
        X,
        theta_value=theta_value,
        metric="euclidean"
    )
    return cents, labels


def berserker_papprox(points, theta_value, metric='euclidean'):
    """
    Parallel approximate berserker clustering using Ray.
    Returns per-chunk centroids (no global merge yet).
    """

    # Ensure contiguous, writeable array before putting into Ray
    X = np.ascontiguousarray(points, dtype=np.float64)
    X.setflags(write=True)

    # Store in Ray object store
    X_id = ray.put(X)

    # Compute chunk size
    n_cpu = int(ray.available_resources().get("CPU", 1))

    chunk_size = max(1, len(X) // n_cpu)

    futures = []
    for i in range(0, len(X), chunk_size):
        end = min(i + chunk_size, len(X))
        futures.append(
            process_chunk.remote(X_id, i, end, theta_value)
        )

    # Retrieve all results
    results = ray.get(futures)   # list of (centroids_i, labels_i)

    # Combine all chunk centroids into one array
    all_centroids = [c for c, _ in results]

    # Optional: labels per chunk also returned
    all_labels = [lbl for _, lbl in results]

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

    X, Ctrue, _ = make_grid_gaussians(A=200, B=50, l=10.0, sigma=1.0, n_per=500)

    np.random.shuffle(X)

    print(X.shape)

    theta = 6

    t0 = time()
    centroids, labels = berserker(X, theta_value=theta, metric="euclidean")
    t1 = time()

    cents, labels = berserker_papprox(X, theta_value=theta, metric='euclidean')
    t2 = time()
    print("Berserker time: %.4f s" % (t1 - t0))
    print("Berserker papprox time: %.4f s" % (t2 - t1))


    print("Number of chunks:", len(cents))
    print("Done.")

    print("Final number of centroids:", len(centroids))
    print("Cents:", len(cents[0]))

    t = 1

    # Merging phase

