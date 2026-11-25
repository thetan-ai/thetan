# Distance metrics

Every Thetan function that takes a `metric` accepts the same names. They
are implemented in C in `thetan.metrics` and oriented so that **a smaller
value always means two rows are closer**. θ is compared with that value.

| Name | Value for rows `a` and `b` |
| --- | --- |
| `euclidean` | `sqrt(sum((a - b)**2))` |
| `ssd` | `sum((a - b)**2)` |
| `norm_ssd` | `mean((a - b)**2)` |
| `norm_ssd_un` | same as `norm_ssd`, with an unrolled loop |
| `l1_norm` | `sum(abs(a - b))` |
| `inf_norm` | `max(abs(a - b))` |
| `cosine` | `1 - cos(a, b)` |
| `arccos` | the angle between `a` and `b`, in radians |
| `dot_prod` | `-dot(a, b)` |

## The negative dot product

`dot_prod` flips the sign of the dot product so that it behaves like a
distance. θ is then negative: `theta=-0.9` means "join a centroid when the
dot product is above 0.9". The dot product also grows with the length of
the vectors, so normalise rows to unit length first; on unit vectors it is
the cosine similarity.

## Computing distances directly

{func}`distance_matrix <thetan.metrics.distance_matrix>` evaluates any of
these metrics between every row of two arrays:

```python
import numpy as np
from thetan import distance_matrix

A = np.ascontiguousarray(np.random.rand(5, 3))
B = np.ascontiguousarray(np.random.rand(4, 3))
D = distance_matrix(A, B, metric="cosine")   # shape (5, 4)
```

## See it

{ref}`sphx_glr_gallery_distance_metrics.py` clusters 3D directions with the
angular metrics and shows why the Euclidean metric fails on them.
