# From TS to TB

{.subtitle}
Two passes. One idea, let the centroids arrive first.

## Thetan Sequential (TS)

TS is the primitive every Thetan algorithm is built from. It visits each
point exactly once:

```text
for x in X:
    μ ← nearest centroid
    if d(x, μ) < θ:  x joins μ, and μ's running mean moves toward x
    else:            x opens a new cluster at x
```

One pass, one parameter, and memory that grows with the number of
clusters rather than with the number of pairs of points. TS works on a
stream.

Its weakness is order. Centroids are still moving while the pass runs, so
a point that comes early can open a cluster *between* two true groups.
Shuffle the input and the partition changes.

```python
from thetan import TS

centroids, labels = TS(X, theta=3.6)
```

## Thetan Berserker (TB)

TB keeps the speed of TS and removes most of its order sensitivity. It
reuses TS on the data, on the centroids, and on both together:

1. **Pass 1.** Shuffle the data and run TS.
2. **Cleanup.** Run TS on the centroids alone. Centroids within θ of each
   other land in one group; the oldest is kept and the others are flagged
   as spurious.
3. **Delete and reassign.** Spurious centroids are deleted, their points
   move to the nearest surviving centroid, and centroids are recomputed
   from the labels.
4. **Pass 2.** Place the cleaned centroids *in front of* a new shuffle of
   the data and run TS again. Every point is now compared with near-final
   centroids instead of whichever points happened to come first.
5. **Second cleanup.** Remove the few spurious centroids that outliers
   create in pass 2, undo the shuffle, and return the centroids and labels.

```python
from thetan import TB

centroids, labels = TB(X, theta=3.6)
```

Steps 2 and 3 are {func}`candidate_labels_c <thetan.core.candidate_labels_c>`,
{func}`reassign_uncertain_update_labels <thetan.core.reassign_uncertain_update_labels>`
and {func}`centroids_from_labels <thetan.core.centroids_from_labels>`.
Step 4 is {func}`sequential_prelim <thetan.core.sequential_prelim>`. You
can call them yourself to build variants.

## See it

- {ref}`sphx_glr_gallery_berserker_gaussian.py` compares the centroids of TS and TB
  on 200 Gaussian clusters.
- {ref}`sphx_glr_gallery_order_robustness.py` clusters the same data in twenty orders
  and measures how much the partitions agree.
