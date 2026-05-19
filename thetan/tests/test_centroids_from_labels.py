import numpy as np
import numpy.testing as npt

from thetan.berserker import (
    X_to_many_3D,
    berserker2_to_3D_plus_extra,
    centroids_from_labels_3d,
    prepare_data_3d,
)

rng = np.random.default_rng(42)

def make_two_cluster_volume(shape=(8, 8, 4), low=50.0, high=200.0, noise=2.0, seed=42):
    rng_local = np.random.default_rng(seed)
    split = shape[0] // 2
    data = np.zeros(shape, dtype=np.float64)
    data[:split, :, :] = low + rng_local.normal(0, noise, (split,) + shape[1:])
    data[split:, :, :] = high + rng_local.normal(0, noise, (shape[0] - split,) + shape[1:])
    return data

def test_single_label_identity_affine():
    data = np.array([[[1.0, 2.0], [3.0, 4.0]],
                     [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    labels = np.zeros((2, 2, 2), dtype=np.int64)
    affine = np.eye(4)

    centroids = centroids_from_labels_3d(data, labels, affine)

    assert centroids.shape == (1, 4)
    npt.assert_allclose(centroids[0], [0.5, 0.5, 0.5, 4.5])

def test_two_labels_identity_affine():
    data = np.array([[[1.0, 2.0], [3.0, 4.0]],
                     [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    labels = np.zeros((2, 2, 2), dtype=np.int64)
    labels[1, :, :] = 1
    affine = np.eye(4)

    centroids = centroids_from_labels_3d(data, labels, affine)

    assert centroids.shape == (2, 4)
    npt.assert_allclose(centroids[0], [0.0, 0.5, 0.5, 2.5])
    npt.assert_allclose(centroids[1], [1.0, 0.5, 0.5, 6.5])

def test_scaling_affine_transforms_coordinates():
    data = np.ones((2, 3, 4), dtype=np.float64)
    labels = np.zeros((2, 3, 4), dtype=np.int64)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    centroids = centroids_from_labels_3d(data, labels, affine)
    assert centroids.shape == (1, 4)
    npt.assert_allclose(centroids[0, :3], [1.0, 3.0, 6.0])


def test_affine_with_translation():
    data = np.ones((2, 2, 2), dtype=np.float64)
    labels = np.zeros((2, 2, 2), dtype=np.int64)
    affine = np.eye(4)
    affine[0, 3] = 10.0
    affine[1, 3] = 20.0
    affine[2, 3] = 30.0

    centroids = centroids_from_labels_3d(data, labels, affine)

    npt.assert_allclose(centroids[0, :3], [10.5, 20.5, 30.5])


def test_centroid_intensity_equals_mean_data():
    shape = (3, 3, 3)
    data = rng.random(shape)
    labels = np.zeros(shape, dtype=np.int64)
    affine = np.eye(4)

    centroids = centroids_from_labels_3d(data, labels, affine)

    npt.assert_allclose(centroids[0, 3], data.mean())


def test_non_contiguous_labels_fills_gap_rows_with_zeros():
    # labels 0, 2, 4 — rows 1 and 3 should be zero
    data = np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float64)
    labels = np.array([[[0]], [[2]], [[4]]], dtype=np.int64)
    affine = np.eye(4)

    centroids = centroids_from_labels_3d(data, labels, affine)

    assert centroids.shape == (5, 4)
    npt.assert_allclose(centroids[0, 3], 1.0)
    npt.assert_allclose(centroids[2, 3], 2.0)
    npt.assert_allclose(centroids[4, 3], 3.0)
    npt.assert_allclose(centroids[1], [0.0, 0.0, 0.0, 0.0])
    npt.assert_allclose(centroids[3], [0.0, 0.0, 0.0, 0.0])


def test_single_voxel_volume():
    data = np.array([[[7.0]]], dtype=np.float64)
    labels = np.array([[[0]]], dtype=np.int64)
    affine = np.eye(4)

    centroids = centroids_from_labels_3d(data, labels, affine)

    assert centroids.shape == (1, 4)
    npt.assert_allclose(centroids[0], [0.0, 0.0, 0.0, 7.0])

def test_matches_berserker2_to_3D_plus_extra():
    data = make_two_cluster_volume(shape=(8, 8, 4), low=50.0, high=200.0,
                                   noise=2.0, seed=42)
    affine = np.eye(4)

    X = prepare_data_3d(data, affine, perc=None, mult=300.0, enh=3.0)
    _, _, _, W = X_to_many_3D(X, data.shape)
    W = np.ascontiguousarray(W, dtype=np.float64)

    theta = 100.0
    data_labels, centroids_berserker, _, _ = berserker2_to_3D_plus_extra(
        X, W, theta
    )

    data_labels_int = np.ascontiguousarray(data_labels, dtype=np.int64)
    assert data_labels_int.ndim == 3
    assert data_labels_int.shape == W.shape

    centroids = centroids_from_labels_3d(W, data_labels_int, affine)

    assert centroids.shape == centroids_berserker.shape
    npt.assert_allclose(centroids, centroids_berserker, atol=1e-5)


def test_berserker_pipeline_finds_two_clusters():
    data = make_two_cluster_volume(shape=(8, 8, 4), low=50.0, high=200.0,
                                   noise=2.0, seed=42)
    affine = np.eye(4)

    X = prepare_data_3d(data, affine, perc=None, mult=300.0, enh=3.0)
    _, _, _, W = X_to_many_3D(X, data.shape)
    W = np.ascontiguousarray(W, dtype=np.float64)

    theta = 100.0
    data_labels, centroids_berserker, _, _ = berserker2_to_3D_plus_extra(
        X, W, theta
    )

    n_unique = len(np.unique(data_labels.astype(np.int64)))
    assert n_unique == 2, f"Expected 2 clusters, got {n_unique}"
    assert centroids_berserker.shape == (2, 4)


if __name__ == "__main__":
    test_single_label_identity_affine()
    test_two_labels_identity_affine()
    test_scaling_affine_transforms_coordinates()
    test_affine_with_translation()
    test_centroid_intensity_equals_mean_data()
    test_non_contiguous_labels_fills_gap_rows_with_zeros()
    test_single_voxel_volume()
    test_matches_berserker2_to_3D_plus_extra()
    test_berserker_pipeline_finds_two_clusters()
    print("All tests passed.")
