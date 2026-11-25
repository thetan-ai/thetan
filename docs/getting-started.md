---
title: Getting started
---

## Install

Thetan is built from source. The Cython extensions need a C compiler
(GCC 8 or later, Clang, or MSVC 2019 or later). On macOS, install
`libomp` with Homebrew to get OpenMP.

```bash
git clone https://github.com/thetan-ai/thetan.git
cd thetan
pip install -r requirements/build.txt
pip install --no-build-isolation .
```

This installs thetan with NumPy as its only requirement. A few modules
need more:

| Module | Needs |
| --- | --- |
| `thetan.tbscan` | scikit-learn |
| `thetan.accuracy` | scipy, scikit-learn |
| `thetan.patchify` | scikit-image |
| `thetan.viz` | matplotlib, scikit-learn |

Install them all with:

```bash
pip install -r requirements/optional.txt
```

## Cluster an array

The clustering functions take a 2D array of shape `(n_samples, n_features)`.
The Cython kernels read it as C-contiguous float64, so convert it first:

```python
import numpy as np
from thetan import TB

rng = np.random.default_rng(0)
centres = np.array([[0, 0], [10, 0], [5, 8]])
points = np.vstack([c + rng.standard_normal((500, 2)) for c in centres])

X = np.ascontiguousarray(points, dtype=np.float64)
centroids, labels = TB(X, theta=4.0)
```

`TB` returns two arrays:

`centroids`
: Shape `(n_clusters, n_features)`. The mean of the points in each cluster.

`labels`
: Shape `(n_samples,)`. The cluster index of each point, in the order of `X`.

Here `len(centroids)` is 3: TB found the three clusters without being told
how many there are.

TB shuffles the data internally with NumPy's global random state. Call
`np.random.seed` first when you need the same result on every run.

## The main functions

| Name | Alias | What it does |
| --- | --- | --- |
| [`berserker`](#api-thetan-berserker-berserker) | `TB` | Two self-correcting passes. The method to use by default. |
| [`sequential`](#api-thetan-core-sequential) | `TS` | One pass. Fast and online, but depends on the order of the data. |
| [`sequential_prelim`](#api-thetan-core-sequential_prelim) | `TP` | One pass seeded with prior centroids. |
| [`tbscan`](#api-thetan-berserker-tbscan) | | TB followed by DBSCAN on the centroids, for non-convex clusters. |
| [`distance_matrix`](#api-thetan-metrics-distance_matrix) | | Any of the metrics between every pair of rows of two arrays. |

All of them are importable from the top-level package:

```python
from thetan import TB, TS, tbscan, distance_matrix
```

## Next steps

- [](./guide/ts-to-tb.md) explains what the two passes do.
- [](./guide/choosing-theta.md) shows how to pick θ.
- The [gallery](./gallery/index.md) has complete, runnable examples.
