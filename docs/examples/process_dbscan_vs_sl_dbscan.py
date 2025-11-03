import os
import time
import numpy as np

import matplotlib.pyplot as plt

from thetan.dbscan import dbscan as thetan_dbscan, DBSCAN_NOISE
from sklearn.cluster import DBSCAN as SKDBSCAN

def make_big_dataset(seed: int = 42, centers: np.ndarray | None = None, per_cluster: int = 900, sigma: float = 0.38, n_noise: int = 300,) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.75, 4.0]], dtype=float)
    
    blobs = [c + sigma * rng.standard_normal((per_cluster, 2)) for c in centers]
    noise = rng.uniform(low=[-1.5,-1.5], high=[7.0,6.0], size=(n_noise,2))
    X = np.vstack([*blobs, noise]).astype(np.float64, copy=False)
    return X

def summarize(labels: np.ndarray, noise_val: int = -1) -> tuple[int, int]:
    labels = np.asarray(labels)
    k = len({int(x) for x in labels if int(x) != noise_val})
    noise_n = int(np.sum(labels == noise_val))
    return k, noise_n

def plot_clusters(ax, X: np.ndarray, labels: np.ndarray, title: str, noise_label: int = -1):
    labels = np.asarray(labels)
    cluster_ids = [cid for cid in sorted(set(labels.tolist())) if cid != noise_label]
    for cid in cluster_ids:
        pts = X[labels == cid]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:,1], s=9, marker='o', label=f'cluster {cid}')
    
    noise_mask = (labels == noise_label)
    if np.any(noise_mask):
        pts = X[noise_mask]
        ax.scatter(pts[:,0], pts[:,1], s=12, marker='x', label='noise')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', frameon=False)

def main():
    X = make_big_dataset()
    eps = 0.60
    min_pts = 5

    t0 = time.perf_counter()
    labels_thetan, k_thetan = thetan_dbscan(X, eps=eps, min_pts=min_pts)
    dt_thetan = time.perf_counter() - t0
    k1, n1 = summarize(labels_thetan, noise_val=DBSCAN_NOISE)

    t1 = time.perf_counter()
    sk = SKDBSCAN(eps=eps, min_samples=min_pts, metric='euclidean')
    labels_sklearn = sk.fit_predict(X)
    dt_sklearn = time.perf_counter() - t1
    k2, n2 = summarize(labels_thetan, noise_val = -1)

    print(
        f"thetan.dbscan: k={k1} clusters, noise={n1}/{len(X)}"
        f"(eps={eps}. min_pts={min_pts}, time={dt_thetan:.3f}s)"
    )

    print(
        f"sklearn.DBSCAN: k={k2} clusters, noise={n2}/{len(X)}"
        f"(eps={eps}, min_pts={min_pts}, time={dt_sklearn:.3f}s)"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), constrained_layout = True)
    plot_clusters(ax1, X, labels_thetan, f"thetan.dbscan (k={k1}, noise={n1})", noise_label=DBSCAN_NOISE)
    plot_clusters(ax2, X, labels_sklearn, f"sklearn.DBSCAN (k={k2}, noise={n2})", noise_label=-1)
    
    plt.show()

if __name__ == "__main__":
    main()