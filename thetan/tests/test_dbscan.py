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