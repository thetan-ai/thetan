import os
import numpy as np
import matplotlib.pyplot as plt

from thetan.dbscan import dbscan, DBSCAN_NOISE

# Dataset with 3 2d gaussian blobs + some uniform noise
def make_toy_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)

    centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]], dtype=float)
    per_cluster = 60
    sigma = 0.35

    A = centers[0] + sigma * rng.standard_normal((per_cluster, 2))
    B = centers[1] + sigma * rng.standard_normal((per_cluster, 2))
    C = centers[2] + sigma * rng.standard_normal((per_cluster, 2))

    noise = rng.uniform(low=[-1.0,-1.0], high=[6.5,5.5], size = (20,2))

    X = np.vstack([A,B,C,noise]).astype(np.float64, copy=False)
    return X

def run_dbscan_and_plot(X, eps=0.6, min_pts=5, save_path="dbscan_clusters.png"):
    labels, k = dbscan(X, eps=eps, min_pts = min_pts)
    noise_mask = labels == DBSCAN_NOISE
    n_noise = int(np.sum(noise_mask))

    print(f"DBSCAN: k={k} clusters, noise={n_noise}/{len(X)} points"
    f"(eps={eps}, min_pts={min_pts})")

    fig, ax = plt.subplots()

    # clusters_ids = sorted(set(labels.tolist()))
    cluster_ids = [cid for cid in sorted(set(labels.tolist())) if cid != DBSCAN_NOISE]
    for cid in cluster_ids:
        mask = labels == cid
        pts = X[mask]
        if len(pts):
            ax.scatter(pts[:,0], pts[:,1], s=24, marker = "o", label=f"cluster {cid}")
    
    if np.any(noise_mask):
        pts = X[noise_mask]
        ax.scatter(pts[:,0], pts[:,1], s=36, marker='x',label="noise")
    
    ax.set_title(f"DBSCAN result (k={k}, noise={n_noise})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    if save_path:
        fig.save_fig(save_path, dpi=300)
    else:
        plt.show()

def main():
    X = make_toy_dataset()
    run_dbscan_and_plot(X, eps = 0.6, min_pts=5, save_path=None)

if __name__ == "__main__":
    main()