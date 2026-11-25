---
title: Non-convex clusters
subtitle: TBSCAN and DBSCAN
---

TB returns compact clusters: every point is within about θ of its
centroid. Rings, moons and spirals are not compact, so TB tiles them with
many small clusters.

## TBSCAN

[`tbscan`](#api-thetan-berserker-tbscan) links those tiles. It runs TB with
a small θ, then DBSCAN on the TB centroids instead of on every point, and
gives each point the DBSCAN label of its centroid.

```python
from thetan import tbscan

centroids, labels = tbscan(X, theta=0.08, eps=0.15, min_samples=1)
```

`theta`
: The TB threshold. Small enough that a tile never spans two shapes.

`eps`
: The DBSCAN radius between centroids. A little larger than θ, so that
  neighbouring tiles of the same shape connect.

`min_samples`
: The number of centroids, not points, a neighbourhood needs to be dense.
  With 1, every centroid is a core point and no point is noise.

Because DBSCAN only sees the centroids, its cost depends on the number of
TB clusters rather than on the number of points.

The returned `centroids` are unweighted means of TB centroids. Points
whose centroid DBSCAN marks as noise get the label -1.

## DBSCAN on its own

`thetan.dbscan` also ships a standalone DBSCAN, compiled with Cython:

```python
from thetan.dbscan import dbscan, DBSCAN_NOISE

labels, n_clusters = dbscan(X, eps=0.5, min_pts=5)
```

It finds neighbours by brute force, so it is best suited to small inputs,
such as the centroids of TB.

## See it

- [](../gallery/tbscan_shapes.md) runs TB and TBSCAN on moons and circles.
- [](../gallery/tbscan_spiral_blobs.md) compares DBSCAN on 30,000 spiral
  points with a density-aware TBSCAN.
