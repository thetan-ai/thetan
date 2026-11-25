"""
Non-convex clusters with TBSCAN
===============================

Thetan Berserker returns compact clusters, so it splits a ring or a moon
into several pieces. TBSCAN links those pieces with DBSCAN run on the TB
centroids instead of on every point, which recovers non-convex shapes at
a fraction of the cost.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

from thetan import TB, tbscan

# %%
# Two datasets
# ------------

np.random.seed(0)
datasets = {
    "moons": make_moons(n_samples=6000, noise=0.06, random_state=0),
    "circles": make_circles(n_samples=6000, factor=0.5, noise=0.04, random_state=0),
}

# %%
# TB alone, then TBSCAN
# ---------------------
#
# With a small θ, TB tiles each shape with many small clusters. DBSCAN then
# joins TB centroids closer than eps, with eps a little larger than θ so
# that neighbouring tiles connect. A few outlying points can still form a
# tiny cluster of their own, which barely moves the ARI.

theta, eps = 0.08, 0.15
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for row, (name, (X, y)) in enumerate(datasets.items()):
    X = np.ascontiguousarray(X, dtype=np.float64)
    tb_centroids, tb_labels = TB(X, theta)
    sc_centroids, sc_labels = tbscan(X, theta, eps=eps, min_samples=1)
    for col, (labels, k, title) in enumerate((
        (tb_labels, len(tb_centroids), "TB"),
        (sc_labels, len(sc_centroids), "TBSCAN"),
    )):
        ari = adjusted_rand_score(y, labels)
        print(f"{name:>7} {title:>6}: {k:3d} clusters, ARI = {ari:.3f}")
        ax = axes[row, col]
        ax.scatter(X[:, 0], X[:, 1], c=np.asarray(labels) % 20, cmap="tab20", s=2)
        ax.set_title(f"{name}, {title}: {k} clusters, ARI {ari:.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
plt.show()
