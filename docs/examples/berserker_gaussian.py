"""
Thetan Sequential vs Thetan Berserker on a grid of Gaussians
============================================================

Cluster 200 Gaussian blobs laid out on a 20 x 10 grid with one distance
threshold, θ, and compare one pass of Thetan Sequential (TS) with the
two self-correcting passes of Thetan Berserker (TB). Neither method is
told how many clusters there are.
"""

import matplotlib.pyplot as plt
import numpy as np

from thetan import TB, TS

# %%
# Simulate the data
# -----------------
#
# Each blob is a unit Gaussian with 200 points. The centres are 4.8 apart,
# so a threshold of θ = 3.6 is larger than a typical cluster and smaller
# than the gap between two neighbours.

rng = np.random.default_rng(42)
np.random.seed(42)

spacing = 4.8
n_x, n_y = 20, 10
n_per_cluster = 200

gx, gy = np.meshgrid(np.arange(n_x) * spacing, np.arange(n_y) * spacing)
centres = np.column_stack([gx.ravel(), gy.ravel()])
X = np.vstack([c + rng.standard_normal((n_per_cluster, 2)) for c in centres])
X = X[rng.permutation(len(X))]
X = np.ascontiguousarray(X, dtype=np.float64)

print(f"{X.shape[0]} points, {X.nbytes / 1024**2:.1f} MB, {len(centres)} true clusters")

# %%
# One pass: Thetan Sequential
# ---------------------------
#
# TS visits each point once. A point joins its nearest centroid when it is
# closer than θ; otherwise it opens a new cluster. It is fast, but a point
# seen early can open a cluster between two true groups.

theta = 3.6
ts_centroids, ts_labels = TS(X, theta)
print(f"TS found {len(ts_centroids)} clusters")

# %%
# Two passes: Thetan Berserker
# ----------------------------
#
# TB removes the duplicate centroids of the first pass, then runs a second
# pass with the cleaned centroids placed in front of a new shuffle of the
# data.

tb_centroids, tb_labels = TB(X, theta)
print(f"TB found {len(tb_centroids)} clusters")

# %%
# Compare the centroids with the true centres
# -------------------------------------------
#
# Red crosses are the centroids each method returns and black circles the
# true centres. The extra centroids of TS sit between blobs; TB's land on
# the centres.

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, centroids, name in ((axes[0], ts_centroids, "Thetan Sequential (TS)"),
                            (axes[1], tb_centroids, "Thetan Berserker (TB)")):
    ax.scatter(X[:, 0], X[:, 1], s=1, c="0.75", rasterized=True)
    ax.scatter(centres[:, 0], centres[:, 1], s=30, facecolors="none",
               edgecolors="k", linewidths=0.8, label="true centre")
    ax.scatter(centroids[:, 0], centroids[:, 1], s=18, c="tab:red",
               marker="x", label="centroid")
    ax.set_title(f"{name}: {len(centroids)} clusters")
    ax.set_aspect("equal")
axes[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.show()
