import numpy as np
import numpy.testing as npt
from thetan import sequential, sequential_prelim, berserker, tbscan
from thetan import centroids_from_labels, candidate_labels_c, reassign_uncertain_update_labels, distance_matrix


rng = np.random.default_rng(42)

def make_grid_gaussians(A=4, B=3, l=5.0, sigma=1.0, n_per=150, d=2):
    xs = np.arange(-(A-1)/2, (A+1)/2) * l
    ys = np.arange(-(B-1)/2, (B+1)/2) * l
    C = np.array([(x,y) for x in xs for y in ys], dtype=float)
    X = np.vstack([c + rng.normal(scale=sigma, size=(n_per,d)) for c in C])
    y = np.repeat(np.arange(len(C)), n_per)
    return X, C, y


def candidate_labels_reference(centroids, labels, initial_labels):
    """Pure-Python reference for candidate_labels_c"""
    nice = np.ones(initial_labels.shape[0], dtype=initial_labels.dtype)
    updated_labels = initial_labels.copy()
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    for c in clusters:
        if len(c) > 1:
            cs = np.sort(c)
            candidates.extend(cs[1:])

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l == can:
                nice[i] = 0
                updated_labels[i] = -1

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l > can:
                updated_labels[i] -= 1

    return nice, candidates, clusters, updated_labels


def match_fraction(estimated, truth, metric='euclidean', eps=0.8):
    D = distance_matrix(estimated, truth, metric=metric)
    closest = D.min(axis=1)
    return float(np.sum(closest < eps)) / len(truth)

def test_sequential_basic_roundtrip():
    X, _, _ = make_grid_gaussians(A=3, B=2, l=6.0, sigma=1.0, n_per=50)
    theta = 3.6
    cents, labels = sequential(X, theta, metric='euclidean', all_out=False)
    cents2 = centroids_from_labels(X, labels)
    f = match_fraction(cents2, cents, metric='euclidean',eps=1e-6)
    assert f == 1.0

def test_sequential_dot_prod_groups_similar_vectors():
    # [1, 0] and [1, 0.1] point the same way, [0, 1] is at a right angle
    X = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    _, labels = sequential(X, -0.5, metric='dot_prod')
    npt.assert_array_equal(labels, [0, 0, 1])

def test_sequential_zero_rows_share_one_cluster():
    # zero rows have no direction: they form one cluster of their own
    # instead of joining a real cluster or each starting a new one
    X = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.1], [0.0, 0.0], [0.0, 0.0]])
    for metric, theta in [('cosine', 0.3), ('arccos', 0.8)]:
        cents, labels = sequential(X, theta, metric=metric)
        npt.assert_array_equal(labels, [0, 1, 0, 1, 1])
        npt.assert_array_equal(cents[1], [0.0, 0.0])

def test_tb_recovers_right_K_on_simple_grid():
    X, Ctrue, _ = make_grid_gaussians(A=3, B=2, l=6.0, sigma=1.0, n_per=60)
    theta = 3.6
    cents, labels = berserker(X, theta, metric='euclidean')
    f = match_fraction(cents, Ctrue, metric='euclidean',eps=0.8)
    assert f >= 0.95, f"TB should recover most true centroids (got {f:.3f})"

def test_candidate_labels_python_vs_cython():
    X, _, _ = make_grid_gaussians(A=3, B=2, l=4.0, n_per=10)
    theta = 3.6
    C0, L0 = sequential(X, theta, metric='euclidean', all_out=False)
    Cc, Lc = sequential(C0, theta, metric='euclidean', all_out=False)
    nice_py, cand_py, _cl_py, upd_py = candidate_labels_reference(Cc, Lc, L0)
    nice_c, cand_c, _cl_c, upd_c = candidate_labels_c(Cc, Lc, L0)
    npt.assert_array_equal(nice_c, nice_py)
    npt.assert_array_equal(np.sort(cand_c), np.sort(np.asarray(cand_py)))
    npt.assert_array_equal(upd_c, upd_py)

def test_reassign_uncertain_has_no_negative_after():
    X, _, _ = make_grid_gaussians(A=2, B=2, l=4.0, n_per=10)
    theta = 3.6
    C0, L0 = sequential(X, theta, metric='euclidean', all_out=False)
    Cc, Lc = sequential(C0, theta, metric='euclidean', all_out=False)
    nice, candidates, _c, updated = candidate_labels_c(Cc, Lc, L0)
    Kept = np.delete(C0, candidates, axis=0)
    reassigned = reassign_uncertain_update_labels(X, Kept, updated, nice)
    assert (reassigned >= 0).all()

def test_tb_less_order_sensitive_than_ts():
    X, Ctrue, _ = make_grid_gaussians(A=3, B=3, l=6.0, n_per=30)
    theta = 3.6
    k_ts = []
    k_tb = []
    for _ in range(10):
        perm = rng.permutation(len(X))
        Xp = X[perm]
        c_ts, _ = sequential(Xp, theta, metric='euclidean', all_out=False)
        c_tb, _labels = berserker(Xp, theta, metric='euclidean')
        k_ts.append(len(c_ts))
        k_tb.append(len(c_tb))
    mad = lambda a: np.median(np.abs(a - np.median(a)))
    assert mad(k_tb) <= mad(k_ts)

def _gaussian_grid(A=5, B=5, blob_dis=5, n_local=200, sigma=1.0, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.arange(-blob_dis * (A//2), blob_dis * ((A+1)//2), blob_dis)
    ys = np.arange(-blob_dis * (B//2), blob_dis * ((B+1)//2), blob_dis)
    gx, gy = np.meshgrid(xs, ys)
    centers = np.column_stack([gx.ravel(), gy.ravel()])

    K = centers.shape[0]
    X = np.zeros((K * n_local, 2), dtype=np.float64)
    gt_centroids = []

    for i,c in enumerate(centers):
        block = c + rng.normal(0.0, sigma, size=(n_local, 2))
        X[i * n_local: (i+1) * n_local] = block
        gt_centroids.append(block.mean(axis=0))

    return X, np.ascontiguousarray(np.vstack(gt_centroids), dtype=np.float64)


def test_berseker_gaussian_grid_recovery():
    X, gt_centroids = _gaussian_grid(A=5, B=5, blob_dis=5, n_local=200, sigma=1.0, seed=0)
    K_true = gt_centroids.shape[0]

    theta_value = 3.6

    np.random.seed(0)
    pred_centroids, pred_labels = berserker(X, theta_value, metric='euclidean')

    npt.assert_equal(pred_labels.shape[0], X.shape[0])
    npt.assert_equal(pred_centroids.shape[1], X.shape[1])

    npt.assert_equal(pred_centroids.shape[0], K_true)

    D = distance_matrix(np.ascontiguousarray(pred_centroids, dtype=np.float64), gt_centroids, metric='euclidean')
    closest = np.min(D, axis=1)

    npt.assert_array_less(closest, 1.0)


def test_tbscan_merges_tb_centroids():
    X, gt_centroids = _gaussian_grid(A=3, B=3, blob_dis=8, n_local=200, sigma=1.0, seed=1)

    np.random.seed(0)
    tb_centroids, _ = berserker(X, 3.6)

    np.random.seed(0)
    centroids, labels = tbscan(X, 3.6, eps=2.0, min_samples=1)

    npt.assert_equal(labels.shape[0], X.shape[0])
    assert labels.min() >= 0
    assert labels.max() == centroids.shape[0] - 1
    assert centroids.shape[0] <= tb_centroids.shape[0]


def test_sequential_point_exactly_theta_away_starts_a_new_cluster():
    # [0, 0] and [3, 4] are exactly 5 apart. A point joins a cluster only
    # when its distance is strictly less than theta.
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    _, labels = sequential(X, 5.0)
    npt.assert_array_equal(labels, [0, 1])
    _, labels = sequential(X, 5.0 + 1e-9)
    npt.assert_array_equal(labels, [0, 0])

def test_sequential_all_out_sums_and_sizes():
    local_rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([c + local_rng.normal(scale=1.0, size=(40, 2)) for c in centers])
    X = X[local_rng.permutation(len(X))]
    cents, labels, sums, sizes = sequential(X, 3.6, all_out=True)
    K = cents.shape[0]
    npt.assert_array_equal(sizes, np.bincount(labels, minlength=K))
    npt.assert_allclose(sums, [X[labels == k].sum(axis=0) for k in range(K)])
    npt.assert_allclose(cents, sums / sizes[:, None])
    # the extra outputs do not change the clustering
    cents2, labels2 = sequential(X, 3.6)
    npt.assert_array_equal(labels2, labels)
    npt.assert_array_equal(cents2, cents)

def test_reassign_uncertain_picks_the_nearest_centroid():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.array([[9.0, 1.0], [1.0, 8.0], [0.5, 0.5], [9.5, 0.2], [1.0, 9.0]])
    # negative labels (not always -1) mark the points to reassign. The last
    # two points keep their label, even though [9.5, 0.2] is closer to
    # centroid 1 than to its own centroid 0.
    labels = np.array([-1, -1, -4, 0, 2], dtype=np.intp)
    nice = np.ones(len(X), dtype=np.intp)
    reassigned = reassign_uncertain_update_labels(X, centroids, labels, nice)
    npt.assert_array_equal(reassigned, [1, 2, 0, 0, 2])

def test_reassign_uncertain_uses_the_metric():
    # [3, 1] is closest to [0, 1] in distance and to [10, 0] in direction
    centroids = np.array([[10.0, 0.0], [0.0, 1.0]])
    X = np.array([[3.0, 1.0]])
    labels = np.array([-1], dtype=np.intp)
    nice = np.ones(1, dtype=np.intp)
    npt.assert_array_equal(reassign_uncertain_update_labels(X, centroids, labels, nice), [1])
    npt.assert_array_equal(reassign_uncertain_update_labels(X, centroids, labels, nice, metric='cosine'), [0])

def test_sequential_prelim_hand_computed():
    # theta = 1.5. The prior centroids [0, 0] and [1, 0] are 1 apart, so
    # they merge into one cluster with mean [0.5, 0]; [10, 0] stays alone.
    means_prev = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    # [0.5, 0] joins the first cluster, [10, 1] joins the second, and
    # [20, 0] is too far from both and starts a third.
    X = np.array([[0.5, 0.0], [10.0, 1.0], [20.0, 0.0]])
    cents, labels, sums, sizes = sequential_prelim(X, means_prev, 1.5, all_out=True)
    # only the points in X get a label, the prior centroids do not
    npt.assert_array_equal(labels, [0, 1, 2])
    # each prior centroid counts as one member of its cluster
    npt.assert_array_equal(sizes, [3, 2, 1])
    npt.assert_allclose(sums, [[1.5, 0.0], [20.0, 1.0], [20.0, 0.0]])
    npt.assert_allclose(cents, [[0.5, 0.0], [10.0, 0.5], [20.0, 0.0]])
    cents2, labels2 = sequential_prelim(X, means_prev, 1.5)
    npt.assert_array_equal(labels2, labels)
    npt.assert_allclose(cents2, cents)

def test_sequential_prelim_uses_the_metric():
    # [3, 1] is closest to [0, 1] in distance and to [10, 0] in direction
    means_prev = np.array([[10.0, 0.0], [0.0, 1.0]])
    X = np.array([[3.0, 1.0]])
    _, labels = sequential_prelim(X, means_prev, 3.5)
    npt.assert_array_equal(labels, [1])
    _, labels = sequential_prelim(X, means_prev, 0.3, metric='cosine')
    npt.assert_array_equal(labels, [0])

def test_berserker_output_contract():
    local_rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    # the blobs are tight compared with theta, so each one is one cluster
    X = np.vstack([c + local_rng.normal(scale=0.5, size=(60, 2)) for c in centers])
    np.random.seed(0)
    cents, labels = berserker(X, 3.6)
    K = cents.shape[0]
    npt.assert_equal(K, 4)
    npt.assert_equal(labels.shape, (X.shape[0],))
    # labels run from 0 to K - 1 and every one of them is used
    npt.assert_array_equal(np.unique(labels), np.arange(K))
    # each centroid is the mean of the points with its label
    npt.assert_allclose(cents, [X[labels == k].mean(axis=0) for k in range(K)])

def test_berserker_with_cosine_groups_by_direction():
    local_rng = np.random.default_rng(0)
    # three directions. The lengths are spread over a factor of 100, so
    # only a metric that ignores length can put each direction together.
    X = np.vstack([(d + 0.03 * local_rng.normal(size=(100, 3))) * local_rng.uniform(1, 100, (100, 1))
                   for d in np.eye(3)])
    direction = np.repeat(np.arange(3), 100)
    np.random.seed(0)
    cents, labels = berserker(X, 0.3, metric='cosine')
    npt.assert_equal(cents.shape[0], 3)
    for g in range(3):
        npt.assert_equal(len(np.unique(labels[direction == g])), 1)

def _rows_of_blobs(sizes=(30, 60, 90), seed=0):
    # two rows of three tight blobs. Blobs in a row are 3 apart and the
    # rows are 30 apart.
    local_rng = np.random.default_rng(seed)
    X, row = [], []
    for r, y in enumerate([0.0, 30.0]):
        for x, n in zip([0.0, 3.0, 6.0], sizes):
            X.append(np.array([x, y]) + local_rng.normal(scale=0.2, size=(n, 2)))
            row.append(np.full(n, r))
    return np.vstack(X), np.concatenate(row)

def test_tbscan_merges_centroids_closer_than_eps():
    X, row = _rows_of_blobs()
    np.random.seed(0)
    tb_centroids, tb_labels = berserker(X, 1.5)
    npt.assert_equal(tb_centroids.shape[0], 6)

    # the three blobs of a row are 3 apart, so eps = 4 chains them together
    np.random.seed(0)
    centroids, labels = tbscan(X, 1.5, eps=4.0)
    npt.assert_equal(centroids.shape[0], 2)
    for r in range(2):
        npt.assert_equal(len(np.unique(labels[row == r])), 1)
    assert labels[row == 0][0] != labels[row == 1][0]
    # each centroid is the plain mean of the three TB centroids it merged,
    # not the mean of the points (the blobs have different sizes)
    for g in range(2):
        merged = np.unique(tb_labels[labels == g])
        npt.assert_equal(len(merged), 3)
        npt.assert_allclose(centroids[g], tb_centroids[merged].mean(axis=0))

    # with eps below the spacing nothing is merged
    np.random.seed(0)
    centroids, labels = tbscan(X, 1.5, eps=2.0)
    npt.assert_equal(centroids.shape[0], 6)

def test_tbscan_min_samples_turns_isolated_centroids_into_noise():
    X, row = _rows_of_blobs()
    far = np.array([100.0, 100.0]) + np.random.default_rng(1).normal(scale=0.2, size=(40, 2))
    X = np.vstack([X, far])
    is_far = np.arange(len(X)) >= len(row)

    # the far blob's centroid has no other centroid within eps
    np.random.seed(0)
    centroids, labels = tbscan(X, 1.5, eps=4.0, min_samples=2)
    npt.assert_equal(centroids.shape[0], 2)
    npt.assert_array_equal(labels[is_far], -1)
    assert labels[~is_far].min() >= 0

    np.random.seed(0)
    centroids, labels = tbscan(X, 1.5, eps=4.0, min_samples=1)
    npt.assert_equal(centroids.shape[0], 3)
    assert labels.min() >= 0


if __name__ == "__main__":
    test_berseker_gaussian_grid_recovery()
