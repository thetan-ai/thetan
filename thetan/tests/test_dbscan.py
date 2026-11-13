import numpy as np
import numpy.testing as npt

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SK_DBSCAN
from sklearn.metrics import adjusted_rand_score

from thetan.dbscan import dbscan

def make_blobs_grid(A=3, B=2, spacing=6.0, sigma=0.30, n_per=100, random_state=0):
    rs = np.random.RandomState(random_state)
    xs = np.arange(A) * spacing
    ys = np.arange(B) * spacing
    centers = np.array([(x,y) for y in ys for x in xs], dtype=float)
    X, _ = make_blobs(n_samples = A * B * n_per, centers=centers, cluster_std=sigma, random_state=rs)
    return np.ascontiguousarray(X, dtype=np.float64), centers

def compare_dbscan_to_sklearn(X, eps, min_samples=5):
    labels_th, K_th = dbscan(X, eps=eps, min_pts=min_samples)
    sk = SK_DBSCAN(eps=eps, min_samples=min_samples)
    sk.fit(X)
    labels_sk = sk.labels_.astype(np.int32, copy=False)
    K_sk = np.unique(labels_sk[labels_sk >= 0]).size

    ari = adjusted_rand_score(labels_sk, labels_th)
    return {
        'ari': ari,
        'K_thetan': int(K_th),
        'K_sklearn': int(K_sk),
        'labels_thetan': labels_th,
        'labels_sklearn': labels_sk,
    }

def test_dbscan_output_contract_and_determinism():
    X, _ = make_blobs_grid(A=2, B=2, spacing=6.0, sigma=0.30, n_per=60, random_state=42)
    eps = 0.80
    min_pts = 5

    labels1, K1 = dbscan(X, eps=eps, min_pts=min_pts)
    labels2, K2 = dbscan(X, eps=eps, min_pts=min_pts)

    assert labels1.shape == (X.shape[0],)
    assert labels1.dtype == np.int32
    assert isinstance(K1, (int, np.integer))

    assert np.all(labels1 >= -1), "Invalid negative labels other than -1 (noise)"
    assert K1 == np.unique(labels1[labels1 >= 0]).size

    npt.assert_array_equal(labels1, labels2)
    assert int(K1) == int(K2)

def test_dbscan_matches_sklearn_on_simple_blobs():
    X, _ = make_blobs_grid(A=3, B=2, spacing=6.0, sigma=0.30, n_per=100, random_state=0)
    eps = 0.80
    min_samples = 5

    out = compare_dbscan_to_sklearn(X, eps, min_samples)
    assert out['K_thetan'] == out['K_sklearn'], f"K mismatch: {out}"
    assert out['ari'] >= 0.99, f"ARI too low: {out['ari']:.3f}"

def test_dbscan_noise_semantics_are_minus_one():
    X, _ = make_blobs_grid(A=2, B=1, spacing=4.0, sigma=0.35, n_per=80, random_state=1)
    eps = 0.35
    min_pts = 10
    labels, K = dbscan(X, eps=eps, min_pts=min_pts)
    assert np.any(labels == -1), "Expected some noise points labelled as -1"
    assert K == np.unique(labels[labels >= 0]).size

def test_dbscan_matches_sklearn_with_noise_points():
    rng = np.random.default_rng(0)
    blobs, _ = make_blobs(n_samples=300, centers=[[0, 0], [8, 0], [0, 8]], cluster_std=0.6, random_state=0)
    scattered = rng.uniform(-4, 12, (60, 2))
    X = np.ascontiguousarray(np.vstack([blobs, scattered]), dtype=np.float64)

    for eps, min_pts in [(0.5, 5), (0.7, 3), (0.9, 10), (0.4, 2)]:
        labels, K = dbscan(X, eps=eps, min_pts=min_pts)
        sk = SK_DBSCAN(eps=eps, min_samples=min_pts).fit(X)
        core = np.zeros(len(X), dtype=bool)
        core[sk.core_sample_indices_] = True

        # same noise points and the same number of clusters
        npt.assert_array_equal(labels == -1, sk.labels_ == -1, err_msg=f'eps={eps}, min_pts={min_pts}')
        npt.assert_equal(K, np.unique(sk.labels_[sk.labels_ >= 0]).size)
        # core points are grouped the same way. A border point between two
        # clusters may go to either one, so those are left out.
        npt.assert_equal(adjusted_rand_score(sk.labels_[core], labels[core]), 1.0)

def test_dbscan_counts_points_exactly_eps_away_as_neighbours():
    # six points in a row, each exactly 1 from the next
    X = np.column_stack([np.arange(6.0), np.zeros(6)])
    labels, K = dbscan(X, eps=1.0, min_pts=2)
    npt.assert_equal(K, 1)
    npt.assert_array_equal(labels, 0)
    npt.assert_array_equal(labels, SK_DBSCAN(eps=1.0, min_samples=2).fit(X).labels_)

def test_dbscan_point_with_exactly_min_pts_neighbours_is_core():
    # the first three points are within eps of each other, so each has
    # exactly 3 neighbours counting itself
    X = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [10.0, 10.0]])
    labels, K = dbscan(X, eps=1.0, min_pts=3)
    npt.assert_array_equal(labels, [0, 0, 0, -1])
    npt.assert_equal(K, 1)
    labels, K = dbscan(X, eps=1.0, min_pts=4)
    npt.assert_array_equal(labels, [-1, -1, -1, -1])
    npt.assert_equal(K, 0)

def test_dbscan_border_point_seen_first_still_joins_the_cluster():
    # the first point has only one neighbour, so it is not a core point and
    # is first marked as noise. It is within eps of [0.3, 0], which is a
    # core point, so the cluster must claim it afterwards.
    X = np.array([[0.75, 0.0], [0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [0.3, 0.3]])
    labels, K = dbscan(X, eps=0.5, min_pts=4)
    npt.assert_array_equal(labels, [0, 0, 0, 0, 0])
    npt.assert_equal(K, 1)
    npt.assert_array_equal(labels, SK_DBSCAN(eps=0.5, min_samples=4).fit(X).labels_)