"""
Clustering directions with angular metrics
==========================================

Thetan works with any metric accepted by
`thetan.distance_matrix`. For data where only the direction of a vector
matters, such as normalised embeddings, the cosine, arccos and negative
dot product metrics give θ a natural meaning: an angle, or a minimum
similarity.
"""

import matplotlib.pyplot as plt
import numpy as np

from thetan import TB, distance_matrix

# %%
# Unit vectors around five directions
# -----------------------------------
#
# Each cluster is a cloud of 3D unit vectors spread a few degrees around
# a mean direction. The vectors are also given random lengths, which an
# angular metric ignores.

rng = np.random.default_rng(3)
np.random.seed(3)

means = rng.standard_normal((5, 3))
means /= np.linalg.norm(means, axis=1, keepdims=True)
spread = np.deg2rad(4)
directions = np.vstack([m + spread * rng.standard_normal((400, 3)) for m in means])
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
lengths = rng.uniform(0.5, 3.0, size=(len(directions), 1))
X = np.ascontiguousarray(directions * lengths)

angles = np.rad2deg(distance_matrix(means, means, metric="arccos"))
print("Smallest angle between two mean directions: "
      f"{angles[np.triu_indices(5, 1)].min():.1f} degrees")

# %%
# One threshold in the units of each metric
# -----------------------------------------
#
# The same partition can be asked for in the units of each metric:
#
# - `arccos`: θ is an angle in radians.
# - `cosine`: θ is `1 - cos(angle)`.
# - `dot_prod`: the negative dot product, so θ is negative. On unit
#   vectors, θ = -0.94 means "join a centroid when the cosine similarity is
#   above 0.94", about 20 degrees. It needs normalised rows, because the
#   dot product grows with vector length.

angle = np.deg2rad(20)
unit = np.ascontiguousarray(directions)
settings = (
    ("arccos", X, angle),
    ("cosine", X, 1 - np.cos(angle)),
    ("dot_prod", unit, -np.cos(angle)),
    ("euclidean", X, 0.4),
)
found = {}
for metric, data, theta in settings:
    centroids, labels = TB(data, theta, metric=metric)
    found[metric] = labels
    print(f"{metric:>9}: θ = {theta:+.3f} -> {len(centroids)} clusters")

# %%
# Why the metric matters
# ----------------------
#
# The angular metrics recover the five directions whatever the vector
# lengths. The Euclidean metric sees vectors of the same direction but
# different lengths as far apart and splits each direction into many
# clusters.

fig = plt.figure(figsize=(11, 5))
for i, metric in enumerate(("arccos", "euclidean")):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    labels = found[metric]
    ax.scatter(*X.T, c=labels % 20, cmap="tab20", s=3)
    ax.set_title(f"{metric}: {len(np.unique(labels))} clusters")
    ax.set_box_aspect((1, 1, 1))
fig.tight_layout()
plt.show()
