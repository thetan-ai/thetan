"""
TB and TBSCAN on a 3-arm spiral
===============================

Thetan Berserker (TB) tiles each arm of a spiral with many small, compact
clusters. TBSCAN, also called TB-DBSCAN, runs DBSCAN on the TB centroids
only and joins those tiles back into the three arms. This example runs TB,
TBSCAN, DBSCAN, HDBSCAN and TB-HDBSCAN on the spiral in
``data/SpiralData_Clean_Spread.csv`` and times DBSCAN against TBSCAN.
"""

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score

from thetan.berserker import berserker
from thetan.viz import plot_centroids_clusters

# sphinx_gallery_thumbnail_number = 4

# %%
# The spiral
# ----------
#
# The CSV holds 7,500 points, 2,500 per arm, in three columns: Chem1, Chem2
# and Clnum. Clnum is the arm each point belongs to, so only the two
# coordinates are clustered and Clnum is kept to score the results.
#
# Stacking copies of the points makes the cost of each method visible.
# ``n_repeats = 10`` gives 75,000 points and runs in a few seconds.
# ``n_repeats = 50`` (375,000 points) is the size of the original TB vs
# TBSCAN figure, but then DBSCAN on every point needs several GB of memory.

n_repeats = 10

try:
    examples_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # sphinx-gallery runs examples from docs/examples, without __file__
    examples_dir = os.getcwd()
fname = os.path.join(examples_dir, "..", "..", "data", "SpiralData_Clean_Spread.csv")

spiral = np.genfromtxt(fname, delimiter=",", skip_header=1)
X = np.ascontiguousarray(np.tile(spiral[:, :2], (n_repeats, 1)))
y = np.tile(spiral[:, 2].astype(int), n_repeats)
print(f"{X.shape[0]:,} points")

np.random.seed(0)
times = {}


def report(name, labels):
    """Print the number of clusters and their agreement with the true arms."""
    n_clusters = len(np.unique(labels[labels >= 0]))
    print(f"{name}: {n_clusters} clusters, ARI = {adjusted_rand_score(y, labels):.2f}")


# %%
# TB
# --
#
# With θ = 1.0, TB cuts each arm into short, compact pieces, about 100
# clusters in all.

theta = 1.0

t = time()
initial_centroids, initial_labels = berserker(X, theta)
times["TB"] = time() - t

report("TB", initial_labels)
plot_centroids_clusters(X, None, initial_labels, title="TB", show=False)

# %%
# DBSCAN on every point
# ---------------------

t = time()
dbscan_labels = DBSCAN(eps=0.5, min_samples=1).fit(X).labels_
times["DBSCAN"] = time() - t

report("DBSCAN", dbscan_labels)
plot_centroids_clusters(X, None, labels=dbscan_labels, title="DBSCAN", show=False)

# %%
# TBSCAN: DBSCAN on the TB centroids only
# ---------------------------------------
#
# DBSCAN now only has to link the TB centroids, and each point then takes
# the label of its centroid. On the two coordinates, eps = 1.9 recovers the
# three arms. The working range is narrow: 1.85 can split an arm and 1.95
# can merge two. :func:`thetan.berserker.tbscan` runs TB and this step in
# one call.

t = time()
centroid_labels = DBSCAN(eps=1.9, min_samples=1).fit(initial_centroids).labels_
tbscan_labels = centroid_labels[initial_labels]
times["TBSCAN"] = time() - t

report("TBSCAN", tbscan_labels)
plot_centroids_clusters(initial_centroids, None, labels=centroid_labels,
                        title="TBSCAN on the TB centroids", show=False)
plot_centroids_clusters(X, None, labels=tbscan_labels, title="TBSCAN", show=False)

# %%
# HDBSCAN, on every point and on the TB centroids
# -----------------------------------------------
#
# With ``min_samples=1``, HDBSCAN on every point treats each group of
# stacked identical copies as a cluster of its own, so it returns one
# cluster per original point. On the TB centroids there are no duplicates,
# and it finds the three arms.

t = time()
hdbscan_labels = HDBSCAN(min_samples=1, copy=True).fit(X).labels_
times["HDBSCAN"] = time() - t

report("HDBSCAN", hdbscan_labels)
plot_centroids_clusters(X, None, labels=hdbscan_labels, title="HDBSCAN", show=False)

t = time()
centroid_hlabels = HDBSCAN(min_samples=1, copy=True).fit(initial_centroids).labels_
tb_hdbscan_labels = centroid_hlabels[initial_labels]
times["TB-HDBSCAN"] = time() - t

report("TB-HDBSCAN", tb_hdbscan_labels)
plot_centroids_clusters(initial_centroids, None, labels=centroid_hlabels,
                        title="TB-HDBSCAN on the TB centroids", show=False)
plot_centroids_clusters(X, None, labels=tb_hdbscan_labels, title="TB-HDBSCAN", show=False)

# %%
# Timings
# -------
#
# TBSCAN's linking step is fast because DBSCAN sees about 100 centroids
# instead of every point. The end-to-end ratio also counts the TB step.

for name, seconds in times.items():
    print(f"{name:>10}: {seconds:7.3f} s")

print(f"DBSCAN / TBSCAN linking step: {times['DBSCAN'] / times['TBSCAN']:.0f}x")
print(f"DBSCAN / (TB + TBSCAN), end to end: {times['DBSCAN'] / (times['TB'] + times['TBSCAN']):.1f}x")

plt.show()
