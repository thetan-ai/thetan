# API reference

Every public function, generated from its numpydoc docstring. The main
entry points are importable from the top-level package:

```python
from thetan import TB, TS, tbscan, distance_matrix
```

| Module | Contents |
| --- | --- |
| [`thetan.berserker`](thetan.berserker.md) | Thetan Berserker (TB), the main clustering method, and TBSCAN, which links TB centroids with DBSCAN to find non-convex clusters. |
| [`thetan.core`](thetan.core.md) | Thetan Sequential (TS), the one-pass primitive, and the cleanup steps that TB builds from it. |
| [`thetan.metrics`](thetan.metrics.md) | Distance functions shared by every Thetan algorithm. |
| [`thetan.dbscan`](thetan.dbscan.md) | A standalone DBSCAN. |
| [`thetan.image`](thetan.image.md) | Helpers that turn images and volumes into feature rows that the clustering functions accept, and post-process segmentations. |
| [`thetan.accuracy`](thetan.accuracy.md) | Scores that compare a clustering with true labels. |
| [`thetan.patchify`](thetan.patchify.md) | Split images into patches and stitch them back. |
| [`thetan.viz`](thetan.viz.md) | Plotting helpers for clustering results. |

```{toctree}
:hidden:

thetan.berserker
thetan.core
thetan.metrics
thetan.dbscan
thetan.image
thetan.accuracy
thetan.patchify
thetan.viz
```
