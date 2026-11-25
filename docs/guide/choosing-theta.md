---
title: Choosing θ
subtitle: θ is the only knob.
---

## The rule

A point joins a centroid when it is closer than θ. So θ sets the scale at
which two points count as being in the same cluster:

- **θ larger than the widest cluster.** Otherwise a cluster is split.
- **θ smaller than the narrowest gap between two clusters.** Otherwise
  neighbours merge.

When both hold, a single pass recovers every cluster with no splits and
no merges ([Theorem 1](./../theory.md#one-to-one-recovery)). In practice,
set θ to the largest diameter you expect a cluster to have.

## When unsure, err small

The two ways of being wrong are not equally bad:

- Too small a θ gives **extra clusters**. They are easy to spot and to merge
  afterwards, for example by running TS on the centroids with a larger θ.
- Too large a θ **merges clusters** for good. No later step separates them.

## θ is in the units of the metric

θ is compared with the value of the distance metric, so its meaning
depends on the metric you pass:

| Metric | θ means |
| --- | --- |
| `euclidean` | a distance, in the units of the data |
| `ssd`, `norm_ssd` | a squared distance (summed or averaged over features) |
| `arccos` | an angle, in radians |
| `cosine` | `1 - cos(angle)` |
| `dot_prod` | minus a dot product, so θ is negative |

See [](./metrics.md) for the full list.

## See it

[](../gallery/choosing_theta.md) sweeps θ over twelve Gaussian clusters
and plots the plateau where TB returns the true number of clusters.
