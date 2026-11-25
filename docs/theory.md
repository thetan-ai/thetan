---
title: Theory
subtitle: Guarantees, not heuristics.
---

The results below are proved in the Thetan Berserker paper
(see [](./citing.md)). The proofs use uniform samples in d-dimensional
balls under the ℓ₂ norm, with extensions to Gaussians in the appendix.
Each step of TB rests on one of them.

(separation-is-preserved)=
## Separation is preserved

**Lemma 1.** If every pair of points from different clusters is at least θ
apart, TS never merges two clusters, whatever the order.

$$\|x_i - x_j\| \ge \theta \implies \text{no cross-cluster merge}$$

(one-to-one-recovery)=
## One-to-one recovery

**Theorem 1.** When θ exceeds every cluster diameter $D_k$ and stays below
every gap ω, a single TS pass returns exactly K clusters: no merges, no
splits.

$$\max_k D_k < \theta \le \omega$$

This is the rule behind [](./guide/choosing-theta.md).

## Short edges vanish with dimension

**Propositions 1 and 2.** For two balls of radius θ/2 whose centres are
3θ/2 apart, the chance that a cross-ball pair lies within θ decays
exponentially in the dimension d. Order mistakes become rarer as the
dimension grows.

$$p_\theta \approx 0.217 \, e^{-0.594 d}$$

## Self-correction in two passes

**Lemma 2 and Theorem 2.** A centroid born from a spurious cross-cluster
merge lies within θ of a true centre, so the second pass absorbs it.

$$\|\mu_m - \nu_1\| \le \tfrac{1}{2}\|y - x\| + \tfrac{\theta}{2} < \theta$$

## Order independence

**Theorems 3 and 4.** For balls of radius r whose centres are more than 4r
apart, the cleanup and the prepended second pass give the same result for
every order.

$$2r < \theta < \ell - 2\varepsilon$$

## Explicit cost bounds

**Theorems 5 and 6.** Time is linear when there is one cluster and
quadratic only when every point is its own cluster. Memory stays linear,
with in-place permutations.

$$O(nd) \;\dots\; O(n^2 d) \text{ time}, \qquad O(n + Kd) \text{ memory}$$
