"""
Density-aware TBSCAN on spirals
===============================

Compare three ways of finding the arms of a 30,000-point spiral: DBSCAN
from thetan on every point, scikit-learn's DBSCAN on every point, and a
density-aware TBSCAN that runs DBSCAN on the centroids of Thetan
Berserker only.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from sklearn.cluster import DBSCAN as SKDBSCAN

from thetan import berserker
from thetan.dbscan import DBSCAN_NOISE, dbscan

# %%
# Helpers
# -------
#
# ``tbscan_density_aware`` weights each TB centroid by the number of points
# it holds and estimates its density from its RMS radius. Weak centroids
# can be pruned, and the DBSCAN parameters on the centroid graph can be
# derived from a target point density.


def ball_volume(r, d):
    """Volume of a d-dimensional ball of radius r."""
    return (np.pi ** (d / 2) / gamma(d / 2 + 1)) * (r ** d)


def tbscan_density_aware(X, initial_centroids, initial_labels, theta, eps=None,
                         min_samples=None, prune_quantile=0.05,
                         use_weight_replication=False, max_replicas=50000):
    """Run DBSCAN on TB centroids, using the points they hold as weights."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    initial_centroids = np.ascontiguousarray(initial_centroids, dtype=np.float64)
    initial_labels = np.asarray(initial_labels, dtype=np.int64)
    N, d = X.shape
    M = initial_centroids.shape[0]

    valid = initial_labels >= 0
    weights = np.bincount(initial_labels[valid], minlength=M).astype(np.float64)

    r = np.full(M, 0.5 * float(theta), dtype=np.float64)
    for label in range(M):
        idx = np.flatnonzero(initial_labels == label)
        if idx.size:
            diffs = X[idx] - initial_centroids[label]
            r[label] = np.sqrt((diffs ** 2).sum(axis=1).mean())

    rho = np.divide(weights, ball_volume(r, d), out=np.zeros_like(weights), where=r > 0)

    keep_w = weights > 0
    keep_rh = rho > 0
    thr_w = np.quantile(weights[keep_w], prune_quantile) if np.any(keep_w) else 0.0
    thr_r = np.quantile(rho[keep_rh], prune_quantile) if np.any(keep_rh) else 0.0
    keep = (weights >= thr_w) & (rho >= thr_r)
    if not np.any(keep):
        keep = weights > 0
        if not np.any(keep):
            keep = np.ones(M, dtype=bool)

    C = initial_centroids[keep]
    w = weights[keep]
    remap = np.full(M, -1, dtype=int)
    remap[np.flatnonzero(keep)] = np.arange(C.shape[0])
    M2 = C.shape[0]

    if eps is None:
        eps = 1.6 * float(theta)

    if min_samples is None:
        rho_keep = rho[keep]
        rho_star = np.median(rho_keep[rho_keep > 0]) if np.any(rho_keep > 0) else 0.0
        if use_weight_replication:
            scale = max(1.0, w.sum() / float(max_replicas))
            min_samples = max(1, int(np.round(rho_star * ball_volume(eps, d) / scale)))
        else:
            typical_w = np.median(w[w > 0]) if np.any(w > 0) else 1.0
            ms = rho_star * ball_volume(eps, d) / typical_w
            min_samples = int(np.clip(np.round(ms), 1, 5))

    if use_weight_replication:
        scale = max(1.0, w.sum() / float(max_replicas))
        reps = np.maximum(1, np.round(w / scale).astype(int))
        C_rep = C[np.repeat(np.arange(M2, dtype=int), reps)]
        dbl_rep, _ = dbscan(np.ascontiguousarray(C_rep), eps=float(eps),
                            min_pts=int(min_samples))
        dbl_rep = np.asarray(dbl_rep, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(reps)[:-1]])
        dbl = dbl_rep[starts]
    else:
        dbl, _ = dbscan(np.ascontiguousarray(C), eps=float(eps), min_pts=int(min_samples))
        dbl = np.asarray(dbl, dtype=np.int64)

    final_labels = np.full(N, DBSCAN_NOISE, dtype=np.int64)
    m = (initial_labels >= 0) & (remap[initial_labels] >= 0)
    final_labels[m] = dbl[remap[initial_labels[m]]]

    cluster_ids = np.unique(final_labels[final_labels != DBSCAN_NOISE])
    info = dict(eps=float(eps), min_samples=int(min_samples), kept=int(M2),
                pruned=int(M - M2), n_clusters=int(cluster_ids.size),
                noise=int(np.sum(final_labels == DBSCAN_NOISE)))
    return final_labels, info


def make_spiral(n_total, arms, rotations, radius, seed):
    """Sample points along a multi-arm spiral, denser near the centre."""
    rng = np.random.default_rng(seed)
    per_arm = n_total // arms
    parts = []
    for k in range(arms):
        t = rng.beta(0.6, 1.2, size=per_arm)
        r = radius[0] + t * (radius[1] - radius[0])
        angle = rotations * 2 * np.pi * t + 2 * np.pi * k / arms
        parts.append(np.column_stack([r * np.cos(angle), r * np.sin(angle)]))
    X = np.vstack(parts)
    return np.ascontiguousarray(X[rng.permutation(len(X))])


def summarize(labels):
    """Count clusters and noise points in a label array."""
    labels = np.asarray(labels)
    return len(set(labels[labels != DBSCAN_NOISE].tolist())), int(np.sum(labels == DBSCAN_NOISE))


def plot_clusters(ax, X, labels, title):
    """Scatter points coloured by cluster, with noise as crosses."""
    labels = np.asarray(labels)
    clustered = labels != DBSCAN_NOISE
    ax.scatter(X[clustered, 0], X[clustered, 1], c=labels[clustered] % 10,
               cmap="tab10", s=2, vmin=0, vmax=9)
    if np.any(~clustered):
        ax.scatter(X[~clustered, 0], X[~clustered, 1], s=6, marker="x", c="k")
    ax.set_title(title)
    ax.set_aspect("equal")


# %%
# The spiral
# ----------
#
# Three arms, 3.2 turns, 30,000 points.

np.random.seed(2025)
X = make_spiral(n_total=30_000, arms=3, rotations=3.2, radius=(0.4, 6.0), seed=2025)
eps, min_pts = 0.5, 1

# %%
# DBSCAN on every point
# ---------------------

t0 = time.perf_counter()
labels_thetan, _ = dbscan(X, eps=eps, min_pts=min_pts)
dt_thetan = time.perf_counter() - t0

t0 = time.perf_counter()
labels_sklearn = SKDBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)
dt_sklearn = time.perf_counter() - t0

print(f"thetan.dbscan : {summarize(labels_thetan)[0]} clusters in {dt_thetan:.2f} s")
print(f"sklearn DBSCAN: {summarize(labels_sklearn)[0]} clusters in {dt_sklearn:.2f} s")

# %%
# TBSCAN: DBSCAN on the TB centroids only
# ---------------------------------------
#
# TB with θ = 0.1 reduces the 30,000 points to about 1,300 centroids.
# DBSCAN then only has to link those, with the same eps as above.

theta = 0.1
t0 = time.perf_counter()
centroids, tb_labels = berserker(X, theta)
dt_tb = time.perf_counter() - t0

t0 = time.perf_counter()
labels_tbscan, info = tbscan_density_aware(
    X, centroids, tb_labels, theta, eps=0.5, min_samples=1,
    prune_quantile=0.0, use_weight_replication=False,
)
dt_link = time.perf_counter() - t0
print(f"TB: {len(centroids)} centroids in {dt_tb:.2f} s")
print(f"TBSCAN: {info['n_clusters']} clusters, linking took {dt_link:.2f} s")

# %%
# Results
# -------

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
plot_clusters(axes[0], X, labels_thetan,
              f"thetan.dbscan: {summarize(labels_thetan)[0]} clusters, {dt_thetan:.2f} s")
plot_clusters(axes[1], X, labels_sklearn,
              f"sklearn DBSCAN: {summarize(labels_sklearn)[0]} clusters, {dt_sklearn:.2f} s")
plot_clusters(axes[2], X, labels_tbscan,
              f"TBSCAN: {info['n_clusters']} clusters, {dt_tb + dt_link:.2f} s")
plt.show()
