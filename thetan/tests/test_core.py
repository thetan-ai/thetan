import numpy as np
import numpy.testing as npt
from thetan import sequential, berserker
from thetan.theta import centroids_from_labels, candidate_labels_c, reassign_uncertain_update_labels, distance_matrix
from thetan.berserker import berserker2_dbg, candidate_labels

rng = np.random.default_rng(42)

def make_grid_gaussians(A=4, B=3, l=5.0, sigma=1.0, n_per=150, d=2):
    xs = np.arange(-(A-1)/2, (A+1)/2) * l
    ys = np.arange(-(B-1)/2, (B+1)/2) * l
    C = np.array([(x,y) for x in xs for y in ys], dtype=float)
    X = np.vstack([c + rng.normal(scale=sigma, size=(n_per,d)) for c in C])
    y = np.repeat(np.arange(len(C)), n_per)
    return X, C, y

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

def test_tb_recovers_right_K_on_simple_grid():
    X, Ctrue, _ = make_grid_gaussians(A=3, B=2, l=6.0, sigma=1.0, n_per=60)
    theta = 3.6
    cents, labels, *_ = berserker2_dbg(X, theta, metric='euclidean', all_out=True, verbose=False)
    f = match_fraction(cents, Ctrue, metric='euclidean',eps=0.8)
    assert f >= 0.95, f"TB should recover most true centroids (got {f:.3f})"

def test_candidate_labels_python_vs_cython():
    X, _, _ = make_grid_gaussians(A=3, B=2, l=4.0, n_per=10)
    theta = 3.6
    C0, L0 = sequential(X, theta, metric='euclidean', all_out=False)
    Cc, Lc = sequential(C0, theta, metric='euclidean', all_out=False)
    nice_py, cand_py, _cl_py, upd_py = candidate_labels(Cc, Lc, L0)
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
        c_tb, _labels, *_ = berserker2_dbg(Xp, theta, metric='euclidean', all_out=True, verbose=False)
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
    npt.assert_equal(pred_centroids.shape[1], X.shape[1])\
    
    npt.assert_equal(pred_centroids.shape[0], K_true)

    D = distance_matrix(np.ascontiguousarray(pred_centroids, dtype=np.float64), gt_centroids, metric='euclidean')
    closest = np.min(D, axis=1)

    npt.assert_array_less(closest, 1.0)