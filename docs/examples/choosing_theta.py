"""
Choosing θ
==========

θ is the only parameter of Thetan Berserker. This example sweeps θ over
twelve Gaussian clusters and shows the plateau where the number of
clusters found matches the truth: θ larger than the widest cluster and
smaller than the narrowest gap between two clusters.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from thetan import TB, TS

# %%
# Twelve unit Gaussians
# ---------------------
#
# The centres are drawn at random, but at least 9 units apart, so that a
# range of θ values separates them.

rng = np.random.default_rng(7)
np.random.seed(7)

centres = []
while len(centres) < 12:
    c = rng.uniform(0, 40, size=2)
    if all(np.linalg.norm(c - other) > 9 for other in centres):
        centres.append(c)
centres = np.array(centres)

n_per_cluster = 300
X = np.vstack([c + rng.standard_normal((n_per_cluster, 2)) for c in centres])
y = np.repeat(np.arange(len(centres)), n_per_cluster)
order = rng.permutation(len(X))
X = np.ascontiguousarray(X[order])
y = y[order]

# %%
# Sweep θ
# -------
#
# For each θ, record how many clusters TS and TB return and how well their
# labels agree with the truth (adjusted Rand index, 1 is a perfect match).

thetas = np.round(np.arange(1.0, 12.01, 0.25), 2)
results = {"TS": ([], []), "TB": ([], [])}
for theta in thetas:
    for name, method in (("TS", TS), ("TB", TB)):
        centroids, labels = method(X, theta)
        results[name][0].append(len(centroids))
        results[name][1].append(adjusted_rand_score(y, labels))

best = thetas[int(np.argmax(results["TB"][1]))]
print(f"Best θ for TB: {best} (ARI {max(results['TB'][1]):.3f})")

# %%
# The plateau
# -----------
#
# Too small a θ splits each cluster into many small ones. Too large a θ
# merges neighbours. In between, TB returns exactly twelve clusters over a
# wide range of θ, and its plateau is wider and flatter than the one of a
# single TS pass.

fig, (ax_k, ax_ari) = plt.subplots(1, 2, figsize=(12, 4))
for name, style in (("TB", "-o"), ("TS", "--s")):
    ax_k.plot(thetas, results[name][0], style, ms=3, label=name)
    ax_ari.plot(thetas, results[name][1], style, ms=3, label=name)
ax_k.axhline(len(centres), color="k", lw=0.8, ls=":", label="true K")
ax_k.set_yscale("log")
ax_k.set_xlabel("θ")
ax_k.set_ylabel("clusters found")
ax_k.legend()
ax_ari.set_xlabel("θ")
ax_ari.set_ylabel("ARI vs truth")
ax_ari.set_ylim(0, 1.05)
ax_ari.legend()
fig.tight_layout()
plt.show()

# %%
# A rule of thumb
# ---------------
#
# Set θ to the largest diameter you expect a cluster to have. When unsure,
# err small: a θ that is too small gives extra clusters, which are easy to
# merge afterwards, while a θ that is too large merges clusters for good.
