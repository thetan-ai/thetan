import numpy as np
from scipy.special import gamma
from thetan.dbscan import dbscan, DBSCAN_NOISE
from thetan import berserker
import time
import matplotlib.pyplot as plt

DBSCAN_NOISE = -1

#Calculating the volume of the n dimensional ball
def Vd(r, d):
    return (np.pi**(d/2) / gamma(d/2 + 1)) * (r**d)

def tbscan_density_aware(
    X,
    initial_centroids,
    initial_labels,
    theta,
    eps=None,
    min_samples=None,
    prune_quantile=0.05,
    use_weight_replication=False,
    max_replicas=50000,
):
    # --- canonical dtypes/shapes ---
    X = np.ascontiguousarray(X, dtype=np.float64)
    initial_centroids = np.ascontiguousarray(initial_centroids, dtype=np.float64)
    initial_labels = np.asarray(initial_labels, dtype=np.int64)
    N, d = X.shape
    M = initial_centroids.shape[0]

    # --- weights per TB centroid (how many original points map to it) ---
    valid = initial_labels >= 0                # keep label 0 as valid
    weights = np.bincount(initial_labels[valid], minlength=M).astype(np.float64)

    # --- empirical RMS radius per centroid (fallback to θ/2) ---
    r = np.full(M, 0.5 * float(theta), dtype=np.float64)
    for l in range(M):
        idx = np.flatnonzero(initial_labels == l)
        if idx.size:
            diffs = X[idx] - initial_centroids[l]
            r[l] = np.sqrt((diffs**2).sum(axis=1).mean())

    # --- density proxy ρ ≈ weight / Vd(radius) ---
    rho = np.divide(weights, Vd(r, d), out=np.zeros_like(weights), where=r > 0)

    # --- prune weak centroids (by weight & density) ---
    keep_w  = weights > 0
    keep_rh = rho > 0
    thr_w = np.quantile(weights[keep_w], prune_quantile) if np.any(keep_w)  else 0.0
    thr_r = np.quantile(rho[keep_rh],    prune_quantile) if np.any(keep_rh) else 0.0
    keep = (weights >= thr_w) & (rho >= thr_r)

    # guard: never drop everything
    if not np.any(keep):
        keep = (weights > 0)
        if not np.any(keep):
            keep = np.ones(M, dtype=bool)

    C = initial_centroids[keep]
    w = weights[keep]
    r = r[keep]
    remap = np.full(M, -1, dtype=int)
    remap[np.flatnonzero(keep)] = np.arange(C.shape[0])
    M2 = C.shape[0]

    # If everything is pruned (paranoid guard), return all-noise
    if M2 == 0:
        final_labels = np.full(N, DBSCAN_NOISE, dtype=np.int64)
        return np.empty((0, d), dtype=X.dtype), final_labels, dict(
            eps=float('nan'), min_samples=0, kept=0, pruned=int(M),
            n_clusters=0, noise=int(final_labels.size)
        )

    # --- choose eps & min_samples on the *centroid* graph ---
    if eps is None:
        eps = 1.6 * float(theta)  # typical for centroids; see TBSCAN examples. :contentReference[oaicite:1]{index=1}

    if min_samples is None:
        # Convert a *raw-point* density target into a *centroid-count* target
        rho_keep = rho[keep]
        rho_star = np.median(rho_keep[rho_keep > 0]) if np.any(rho_keep > 0) else 0.0

        if use_weight_replication:
            # Replication scale: total raw points -> total replicated points
            tot = w.sum()
            scale = max(1.0, tot / float(max_replicas))
            # In replicated space, neighbor weight sum ≈ (raw weight)/scale
            ms = rho_star * Vd(eps, d) / scale
            min_samples = max(1, int(np.round(ms)))
        else:
            # In unweighted centroid space, divide by typical mass per centroid
            typical_w = np.median(w[w > 0]) if np.any(w > 0) else 1.0
            ms = rho_star * Vd(eps, d) / typical_w
            # Clamp: centroids are sparse; 1–5 works best in practice and in the paper. :contentReference[oaicite:2]{index=2}
            min_samples = int(np.clip(np.round(ms), 1, 5))

    # --- DBSCAN on centroids (Cython core) ---
    if use_weight_replication:
        tot = w.sum()
        scale = max(1.0, tot / float(max_replicas))
        reps = np.maximum(1, np.round(w / scale).astype(int))
        idxs = np.repeat(np.arange(M2, dtype=int), reps)
        C_rep = C[idxs]
        dbl_rep, _ = dbscan(np.ascontiguousarray(C_rep, dtype=np.float64),
                                   eps=float(eps), min_pts=int(min_samples))
        dbl_rep = np.asarray(dbl_rep, dtype=np.int64)
        # Condense to one label per centroid group
        dbl = np.empty(M2, dtype=np.int64)
        start = 0
        for j, rj in enumerate(reps):
            dbl[j] = dbl_rep[start]
            start += rj
    else:
        dbl, _ = dbscan(np.ascontiguousarray(C, dtype=np.float64),
                               eps=float(eps), min_pts=int(min_samples))
        dbl = np.asarray(dbl, dtype=np.int64)

    # --- propagate centroid labels back to original points (respect pruning) ---
    final_labels = np.full(N, DBSCAN_NOISE, dtype=np.int64)
    m = (initial_labels >= 0) & (remap[initial_labels] >= 0)
    final_labels[m] = dbl[remap[initial_labels[m]]]

    # --- compute cluster centroids from original X (exclude noise) ---
    cluster_ids = np.unique(final_labels[final_labels != DBSCAN_NOISE])
    if cluster_ids.size:
        centroids = np.vstack([X[final_labels == cid].mean(axis=0) for cid in cluster_ids])
    else:
        centroids = np.empty((0, d), dtype=X.dtype)

    info = dict(
        eps=float(eps),
        min_samples=int(min_samples),
        kept=int(M2),
        pruned=int(M - M2),
        n_clusters=int(cluster_ids.size),
        noise=int(np.sum(final_labels == DBSCAN_NOISE)),
    )
    return centroids, final_labels, info

def make_big_dataset(seed: int = 42,
                     centers: np.ndarray | None = None,
                     per_cluster: int = 900,
                     sigma: float = 0.38,
                     n_noise: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.75, 4.0]], dtype=float)
    blobs = [c + sigma * rng.standard_normal((per_cluster, 2)) for c in centers]
    noise = rng.uniform(low=[-1.5, -1.5], high=[7.0, 6.0], size=(n_noise, 2))
    X = np.vstack([*blobs, noise]).astype(np.float64, copy=False)
    return X

def summarize(labels: np.ndarray, noise_val: int = -1) -> tuple[int, int]:
    labels = np.asarray(labels)
    k = len({int(x) for x in labels if int(x) != noise_val})
    noise_n = int(np.sum(labels == noise_val))
    return k, noise_n

def plot_clusters(ax, X: np.ndarray, labels: np.ndarray, title: str, noise_label: int = -1):
    labels = np.asarray(labels)
    cluster_ids = [cid for cid in sorted(set(labels.tolist())) if cid != noise_label]
    for cid in cluster_ids:
        pts = X[labels == cid]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=9, marker='o', label=f'cluster {cid}')
    noise_mask = (labels == noise_label)
    if np.any(noise_mask):
        pts = X[noise_mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, marker='x', label='noise')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    # ax.legend(loc='best', frameon=False)

def make_spiral_benchmark(
    n_total=300_000,
    arms=3,
    rotations=3.0,
    radius=(0.5, 5.0),
    noise=0.06,
    noise_frac=0.10,
    density='radial_decay',
    seed=7
):
    """
    Generate a large 2D multi-arm spiral dataset with optional background noise.
    Returns (X, y) where X.shape = (n_total, 2), y in {0..arms-1, -1 for noise}.
    """
    rng = np.random.default_rng(seed)
    n_signal = int(n_total * (1 - noise_frac))
    per_arm = n_signal // arms
    remainder = n_signal - per_arm * arms

    X_parts = []
    y_parts = []

    for k in range(arms):
        m = per_arm + (1 if k < remainder else 0)

        # t controls radial position along each arm
        if density == 'radial_decay':
            # more points near the center, sparser outward (hard for density methods)
            t = rng.beta(0.6, 1.2, size=m)
        elif density == 'radial_increase':
            # fewer near center, more outward
            t = 1.0 - rng.beta(0.6, 1.2, size=m)
        else:
            t = rng.random(m)

        r = radius[0] + t * (radius[1] - radius[0])
        angle = rotations * 2 * np.pi * t + (2 * np.pi * k) / arms

        x = r * np.cos(angle) + noise * rng.standard_normal(m)
        y = r * np.sin(angle) + noise * rng.standard_normal(m)

        X_parts.append(np.column_stack([x, y]))
        y_parts.append(np.full(m, k, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # uniform background noise
    n_noise = n_total - X.shape[0]
    if n_noise > 0:
        rmax = radius[1] + 3 * noise
        bound = float(rmax + 0.5)
        ux = rng.uniform(-bound, bound, size=n_noise)
        uy = rng.uniform(-bound, bound, size=n_noise)
        X = np.vstack([X, np.column_stack([ux, uy])])
        y = np.concatenate([y, np.full(n_noise, -1, dtype=np.int64)])

    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def main():
    # X = make_big_dataset()
    X, y = make_spiral_benchmark(
    n_total=30_000,
    arms=3,
    rotations=3.2,
    radius=(0.4, 6.0),
    noise=0.0,
    noise_frac=0.0,
    density='radial_decay',
    seed=2025
    )
    eps = 0.50
    min_pts = 1

    # 1) thetan.dbscan on full data
    t0 = time.perf_counter()
    labels_thetan, k_thetan = dbscan(X, eps=eps, min_pts=min_pts)
    dt_thetan = time.perf_counter() - t0
    k1, n1 = summarize(labels_thetan, noise_val=DBSCAN_NOISE)

    # 2) sklearn.DBSCAN on full data
    t1 = time.perf_counter()
    from sklearn.cluster import DBSCAN as SKDBSCAN
    sk = SKDBSCAN(eps=eps, min_samples=min_pts, metric='euclidean')
    labels_sklearn = sk.fit_predict(X)
    dt_sklearn = time.perf_counter() - t1
    k2, n2 = summarize(labels_sklearn, noise_val=-1)

    # 3) TBSCAN: TB (theta) -> DBSCAN on TB centroids -> propagate
    theta = 0.2
    # TB step
    ttb0 = time.perf_counter()
    initial_centroids, initial_labels = berserker(X, theta)
    dt_tb = time.perf_counter() - ttb0

    # TBSCAN step (density-aware)
    ttb1 = time.perf_counter()
    _, labels_tbscan, info = tbscan_density_aware(
        X, initial_centroids, initial_labels, theta,
        eps=0.5,               # let the function choose eps ≈ 1.6·theta
        min_samples=1,       # let it infer from density proxy
        prune_quantile=0.0,
        use_weight_replication=True
    )
    dt_tbscan = time.perf_counter() - ttb1
    dt_tb_total = dt_tb + dt_tbscan
    k3, n3 = info["n_clusters"], info["noise"]

    # ---- Print summary ----
    print(
        f"thetan.dbscan:   k={k1} clusters, noise={n1}/{len(X)} "
        f"(eps={eps}, min_pts={min_pts}, time={dt_thetan:.3f}s)"
    )
    print(
        f"sklearn.DBSCAN:  k={k2} clusters, noise={n2}/{len(X)} "
        f"(eps={eps}, min_pts={min_pts}, time={dt_sklearn:.3f}s)"
    )
    print(
        f"TBSCAN (TB + thetan.dbscan on centroids): "
        f"k={k3} clusters, noise={n3}/{len(X)} "
        f"(theta={theta}, eps_c≈{1.6*theta:.2f}, time_total={dt_tb_total:.3f}s "
        f"[TB={dt_tb:.3f}s + TBSCAN={dt_tbscan:.3f}s])"
    )

    # ---- Plots ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    plot_clusters(ax1, X, labels_thetan,  f"thetan.dbscan (k={k1}, noise={n1})", noise_label=DBSCAN_NOISE)
    plot_clusters(ax2, X, labels_sklearn, f"sklearn.DBSCAN (k={k2}, noise={n2})", noise_label=-1)
    plot_clusters(ax3, X, labels_tbscan,  f"TBSCAN (k={k3}, noise={n3})")
    plt.show()


if __name__ == "__main__":
    main()