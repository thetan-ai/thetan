# Thetan documentation

{.subtitle}
One distance in. Every cluster out.

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

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc

Build thetan from source and cluster your first array.
:::

:::{grid-item-card} From TS to TB
:link: guide/ts-to-tb
:link-type: doc

How one sequential pass works, why it depends on the order of the data,
and how the second pass fixes it.
:::

:::{grid-item-card} Choosing θ
:link: guide/choosing-theta
:link-type: doc

The rule for picking the only parameter, and what happens when it is too
small or too large.
:::

:::{grid-item-card} Gallery
:link: gallery/index
:link-type: doc

Worked examples with their code, output and figures.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public function, from its numpydoc docstring.
:::

:::{grid-item-card} Theory
:link: theory
:link-type: doc

The guarantees behind each step of TB.
:::

::::

```{toctree}
:hidden:

getting-started
User guide <guide/index>
theory
gallery/index
api/index
development
citing
```
