import os
import math
import numpy as np
import ray
from thetan import berserker, assign_pexact, centroids_from_labels
from time import time
from thetan.viz import plot_centroids_clusters
# from memory_profiler import memory_usage

# (A) Prevent oversubscription (helps a lot if berserker/NumPy uses threads)
THREAD_ENVS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

ray.init(
    include_dashboard=False,
    log_to_driver=False,
    runtime_env={"env_vars": THREAD_ENVS},
)

@ray.remote(num_cpus=1)
def process_chunk(X, start, end, theta_value, metric, return_labels):
    X_chunk = np.require(X[start:end],dtype=np.float64,requirements=["C","W"])

    cents, labels = berserker(X_chunk, theta=theta_value, metric=metric)

    if not return_labels:
        return cents

    labels = labels.astype(np.int32, copy=False)
    return cents, labels

def berserker_papprox(
    points,
    theta_value,
    metric="euclidean",
    *,
    dtype=np.float64,
    chunks_per_cpu=2,
    target_chunk_mb=64,
    return_labels=False,
):
    # Convert once (half memory if float32)
    X = np.asarray(points, dtype=dtype, order="C")
    n = X.shape[0]

    X_ref = ray.put(X)

    # If you DON'T need X locally after this call, uncomment to save one full copy of X in RAM:
    # del X

    n_cpu = int(ray.cluster_resources().get("CPU", 1))
    n_chunks = max(1, n_cpu * chunks_per_cpu)

    bytes_per_row = (np.dtype(dtype).itemsize) * X.shape[1]
    chunk_by_mem = max(1, int((target_chunk_mb * 1024**2) // bytes_per_row))
    chunk_by_cpu = math.ceil(n / n_chunks)
    chunk_size = max(1, min(chunk_by_cpu, chunk_by_mem))

    futures = [
        process_chunk.remote(X_ref, i, min(i + chunk_size, n), theta_value, metric, return_labels)
        for i in range(0, n, chunk_size)
    ]

    # Stream results so you don’t hold a giant “results” list (lower peak memory)
    all_centroids = []
    all_labels = [] if return_labels else None

    while futures:
        done, futures = ray.wait(futures) #, num_returns=min(8, len(futures)))
        for ref in done:
            out = ray.get(ref)
            if return_labels:
                c, lbl = out
                all_centroids.append(c)
                all_labels.append(lbl)
            else:
                all_centroids.append(out)

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

    print('Create dataset ...')
    X, Ctrue, _ = make_grid_gaussians(A=300, B=20, l=10.0, sigma=1.0, n_per=500)

    # np.random.shuffle(X)

    print(X.shape)

    theta = 6

    t0 = time()
    centroids, labels = berserker(X, theta=theta, metric="euclidean")
    t1 = time()
    print("Berserker time: %.4f s" % (t1 - t0))
    print("Berserker number of centroids:", centroids.shape[0])

    # mem_usage = memory_usage(
    # (berserker, (X, theta), {'metric': 'euclidean'}),
    # interval=0.1,
    # max_usage=True
    # )
    # print(f"Peak memory of original berserker: {mem_usage:.2f} MB")


    print('Starting Berserker papprox ...')
    cents, labels = berserker_papprox(X, theta_value=theta, metric='euclidean')
    t2 = time()
    # mem_usage = memory_usage(
    # (berserker_papprox, (X, theta), {'metric': 'euclidean'}),
    # interval=0.1,
    # max_usage=True
    # )
    # print(f"Peak memory of berserker papprox: {mem_usage:.2f} MB")
    # print("Berserker time: %.4f s" % (t1 - t0))
    print("Berserker papprox time: %.4f s" % (t2 - t1))


    print("Number of chunks:", len(cents))
    print("Done.")

    # print("Final number of centroids:", len(centroids))
    # print("Cents:", len(cents[0]))

    t = 1

    ray.shutdown()

    t3 = time()

    # Second phase

    all_cents = np.vstack(cents)
    t4 = time()

    final_cents, final_labels = berserker(all_cents, theta=theta, metric='euclidean')
    t5 = time()
    print("After merging, number of centroids:", len(final_cents))

    os.environ["OMP_NUM_THREADS"] = "12"


    labels = assign_pexact(X, final_cents, metric='euclidean')
    t6 = time()
    centroids = centroids_from_labels(X, labels)
    t7 = time()
    print("Recomputed centroids shape:", centroids.shape)

    print(f"Phase times: papprox {t2 - t1:.4f} s, shutdown ray {t3 - t2:.4f}, s gather {t4 - t3:.4f} s, merge {t5 - t4:.4f} s, assign {t6 - t5:.4f} s, recompute cents {t7 - t6:.4f} s")

    sum = t2 - t1 +t3 - t2 + t4 - t3 + t5 - t4 + t6 - t5 + t7 - t6
    print(f"Total papprox time: {sum:.4f} s")

    # plot_centroids_clusters(X, centroids, labels, title="berserker papprox final clusters")
