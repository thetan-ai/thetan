"""
TBK: K-means seeded by Thetan Berserker
=======================================

K-means needs the number of clusters and a set of starting centres, and
from a poor start it settles with centres in the wrong places. TBK runs
Thetan Berserker (TB) first and starts K-means from the centroids of the
largest TB clusters. This example compares TBK with K-means started by
k-means++ on 100 Gaussian blobs and 20 far-away outliers.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from thetan import TB
from thetan.berserker import tbk

# %%
# Simulate the data
# -----------------
#
# 100 unit Gaussians of 200 points on a 10 x 10 grid, 4.8 apart, the same
# spacing as in the TS vs TB example. 20 single points, evenly spread
# around a circle, lie 40 to 60 units from the middle of the grid, far
# from every blob and from each other.

rng = np.random.default_rng(42)
np.random.seed(42)

spacing = 4.8
n_x, n_y = 10, 10
n_per_cluster = 200
n_outliers = 20

gx, gy = np.meshgrid(np.arange(n_x) * spacing, np.arange(n_y) * spacing)
centres = np.column_stack([gx.ravel(), gy.ravel()])
blobs = np.vstack([c + rng.standard_normal((n_per_cluster, 2)) for c in centres])

angle = np.linspace(0, 2 * np.pi, n_outliers, endpoint=False)
radius = rng.uniform(40, 60, n_outliers)
outliers = centres.mean(axis=0) + np.column_stack([radius * np.cos(angle),
                                                   radius * np.sin(angle)])

X = np.ascontiguousarray(np.vstack([blobs, outliers]), dtype=np.float64)
y = np.concatenate([np.repeat(np.arange(len(centres)), n_per_cluster),
                    np.full(n_outliers, -1)])
in_blob = y >= 0
n_clusters = len(centres)

print(f"{X.shape[0]} points: {n_clusters} blobs and {n_outliers} outliers")

# %%
# What TB finds
# -------------
#
# With θ = 3.6, TB makes one cluster of each blob and one of each outlier.
# TBK keeps only the ``n_clusters`` largest TB clusters as starting
# centres, so the outliers never become centres of their own.

theta = 3.6
tb_centroids, tb_labels = TB(X, theta)
sizes = np.bincount(tb_labels)
print(f"TB found {len(tb_centroids)} clusters: "
      f"{np.sum(sizes > 1)} with {sizes[sizes > 1].min()} to {sizes.max()} points, "
      f"{np.sum(sizes == 1)} with a single point")

# %%
# TBK against K-means with k-means++ starts
# -----------------------------------------
#
# All three methods are given the true number of clusters. A blob counts
# as found when a returned centre lies within 1 unit of its true centre,
# and a centre is away from the blobs when it is further than the grid
# spacing from every true centre. The ARI is computed on the blob points
# only, and the inertia is the K-means cost: the sum of squared distances
# from each point to its centre.


def blobs_found(cluster_centres):
    distances = np.linalg.norm(centres[:, None, :] - cluster_centres[None, :, :], axis=2)
    return distances.min(axis=1) < 1.0


def centres_away_from_blobs(cluster_centres):
    distances = np.linalg.norm(cluster_centres[:, None, :] - centres[None, :, :], axis=2)
    return int(np.sum(distances.min(axis=1) > spacing))


def inertia(cluster_centres, labels):
    return float(np.sum((X - cluster_centres[labels]) ** 2))


results = {}
for name, n_init in (("K-means, 1 start", 1), ("K-means, 10 starts", 10)):
    t = time.perf_counter()
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init,
                   random_state=0).fit(X)
    results[name] = (model.cluster_centers_, model.labels_, time.perf_counter() - t)

t = time.perf_counter()
tbk_centres, tbk_labels = tbk(X, theta, n_clusters)
results["TBK"] = (tbk_centres, tbk_labels, time.perf_counter() - t)

for name, (cluster_centres, labels, seconds) in results.items():
    ari = adjusted_rand_score(y[in_blob], labels[in_blob])
    print(f"{name:>18}: {blobs_found(cluster_centres).sum():3d} of {n_clusters} blobs found, "
          f"{centres_away_from_blobs(cluster_centres)} centres away from the blobs, "
          f"ARI = {ari:.3f}, inertia = {inertia(cluster_centres, labels):,.0f}, "
          f"{seconds:.2f} s")

# %%
# From k-means++ starts, K-means sends a few centres out to the outliers,
# each covering several of them, and leaves others between two blobs. Ten
# starts do not fix this. TBK starts with one centre on each blob, finds
# every blob, and ends with a lower inertia than the best of the ten
# k-means++ runs.
#
# Where the centres land
# ----------------------
#
# Black circles are the true centres and red crosses the returned centres;
# a blob without a centre within 1 unit is filled orange. The plots show
# the grid only, so the centres K-means sent to the outliers are outside
# them.

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, layout="constrained")
for ax, (name, (cluster_centres, labels, seconds)) in zip(axes, results.items()):
    found = blobs_found(cluster_centres)
    ax.scatter(blobs[:, 0], blobs[:, 1], s=1, c="0.8", rasterized=True)
    ax.scatter(centres[~found, 0], centres[~found, 1], s=60, c="tab:orange",
               label="blob not found")
    ax.scatter(centres[:, 0], centres[:, 1], s=30, facecolors="none",
               edgecolors="k", linewidths=0.8, label="true centre")
    ax.scatter(cluster_centres[:, 0], cluster_centres[:, 1], s=18, c="tab:red",
               marker="x", label="returned centre")
    ax.set_title(f"{name}: {found.sum()} of {n_clusters} blobs")
    ax.set_xlim(-4, (n_x - 1) * spacing + 4)
    ax.set_ylim(-4, (n_y - 1) * spacing + 4)
    ax.set_aspect("equal")
axes[0].legend(loc="upper right", fontsize=8)
plt.show()

# %%
# θ must split the data finely enough
# -----------------------------------
#
# TBK needs TB to find at least ``n_clusters`` clusters. A θ larger than
# the gap between blobs merges them, and TBK then raises an error instead
# of starting K-means from too few centres.

try:
    tbk(X, 10.0, n_clusters)
except ValueError as error:
    print(error)
