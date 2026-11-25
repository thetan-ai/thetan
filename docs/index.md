---
title: Thetan documentation
subtitle: One distance in. Every cluster out.
---

Thetan is a family of clustering algorithms built around a single distance
threshold, θ. Its main method, **Thetan Berserker (TB)**, reads the data
twice, corrects its own mistakes and returns the number of clusters it
found. You never pass a cluster count.

```python
import numpy as np
from thetan import TB

X = np.ascontiguousarray(points, dtype=np.float64)   # (n, d)
centroids, labels = TB(X, theta=3.6)                 # K is inferred
print(f"{len(centroids)} clusters")
```

The kernels are written in Cython, run on a single CPU thread and need
only NumPy at runtime.

::::{grid} 1 2 2 2

:::{card} Getting started
:link: ./getting-started.md
Build thetan from source and cluster your first array.
:::

:::{card} From TS to TB
:link: ./guide/ts-to-tb.md
How one sequential pass works, why it depends on the order of the data,
and how the second pass fixes it.
:::

:::{card} Choosing θ
:link: ./guide/choosing-theta.md
The rule for picking the only parameter, and what happens when it is too
small or too large.
:::

:::{card} Gallery
:link: ./gallery/index.md
Worked examples with their code, output and figures.
:::

:::{card} API reference
:link: ./api/index.md
Every public function, from its numpydoc docstring.
:::

:::{card} Theory
:link: ./theory.md
The guarantees behind each step of TB.
:::

::::
