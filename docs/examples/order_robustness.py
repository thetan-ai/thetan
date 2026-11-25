"""
Robustness to the order of the data
===================================

A single sequential pass depends on the order in which it reads the
data. This example clusters the same 10 x 10 grid of Gaussians in twenty
random orders and measures how much consecutive partitions agree, for
Thetan Sequential (TS) and Thetan Berserker (TB).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from thetan import TB, TS

# %%
# The data
# --------
#
# A 10 x 10 grid of unit Gaussians, 5 units apart, clustered with θ = 3.6.

rng = np.random.default_rng(0)
np.random.seed(0)

gx, gy = np.meshgrid(np.arange(10) * 5.0, np.arange(10) * 5.0)
centres = np.column_stack([gx.ravel(), gy.ravel()])
X = np.vstack([c + rng.standard_normal((150, 2)) for c in centres])
theta = 3.6

# %%
# Twenty orders
# -------------
#
# Each run reads a new permutation of X. Labels are mapped back to the
# original order so that runs can be compared point by point.

n_orders = 20
runs = {"TS": [], "TB": []}
for _ in range(n_orders):
    order = rng.permutation(len(X))
    inverse = np.argsort(order)
    Xo = np.ascontiguousarray(X[order])
    for name, method in (("TS", TS), ("TB", TB)):
        centroids, labels = method(Xo, theta)
        runs[name].append((len(centroids), np.asarray(labels)[inverse]))

for name in runs:
    ks = [k for k, _ in runs[name]]
    print(f"{name}: {np.mean(ks):.1f} ± {np.std(ks):.1f} clusters (truth: 100)")

# %%
# Agreement between consecutive orders
# ------------------------------------
#
# The adjusted Rand index (ARI) between two runs is 1 when both orders give
# the same partition. TB's second pass compares every point with
# near-final centroids, which removes most of the order-induced
# disagreement.

agreement = {
    name: [adjusted_rand_score(a[1], b[1]) for a, b in zip(runs[name], runs[name][1:])]
    for name in runs
}
for name, values in agreement.items():
    print(f"{name}: mean ARI between consecutive orders = {np.mean(values):.3f}")

fig, (ax_k, ax_ari) = plt.subplots(1, 2, figsize=(11, 3.8))
ax_k.boxplot([[k for k, _ in runs["TS"]], [k for k, _ in runs["TB"]]])
ax_k.set_xticks([1, 2], ["TS", "TB"])
ax_k.axhline(100, color="k", lw=0.8, ls=":")
ax_k.set_ylabel("clusters found")
ax_ari.boxplot([agreement["TS"], agreement["TB"]])
ax_ari.set_xticks([1, 2], ["TS", "TB"])
ax_ari.set_ylabel("ARI between consecutive orders")
fig.tight_layout()
plt.show()
