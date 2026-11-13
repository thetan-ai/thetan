import numpy as np
import numpy.testing as npt

from scipy import ndimage

from thetan.image import connected_components


def same_partition(labels_a, labels_b):
    """True when two label volumes split the foreground into the same groups."""
    if not np.array_equal(labels_a > 0, labels_b > 0):
        return False
    fg = labels_a > 0
    pairs = np.unique(np.stack([labels_a[fg], labels_b[fg]]), axis=1)
    return len(np.unique(pairs[0])) == pairs.shape[1] == len(np.unique(pairs[1]))


def test_connected_components_output_contract():
    img = np.zeros((4, 5, 6))
    img[0, 0, 0] = 1.0
    img[3, 4, 5] = 1.0

    labels, n = connected_components(img)

    npt.assert_equal(n, 2)
    npt.assert_equal(labels.shape, img.shape)
    npt.assert_equal(labels.dtype, np.dtype(np.intp))
    npt.assert_(labels[0, 0, 0] > 0)
    npt.assert_(labels[3, 4, 5] > 0)
    npt.assert_(labels[0, 0, 0] != labels[3, 4, 5])
    # background stays 0
    npt.assert_equal(labels[img == 0], 0)


def test_connected_components_diagonal_neighbours_join():
    # 26-connectivity: voxels that only touch at a corner are one component
    img = np.zeros((3, 3, 3))
    img[0, 0, 0] = img[1, 1, 1] = img[2, 2, 2] = 1.0

    labels, n = connected_components(img)

    npt.assert_equal(n, 1)
    npt.assert_equal(len(np.unique(labels[img > 0])), 1)


def test_connected_components_matches_scipy_label():
    rng = np.random.default_rng(0)
    full = np.ones((3, 3, 3))
    for _ in range(40):
        shape = tuple(rng.integers(1, 12, size=3))
        img = (rng.random(shape) < rng.uniform(0.05, 0.6)).astype('f8')

        labels, n = connected_components(img)
        ref, n_ref = ndimage.label(img, structure=full)

        npt.assert_equal(n, n_ref)
        npt.assert_(same_partition(labels, ref))


def test_connected_components_delta():
    # two touching blocks whose intensities differ by 4
    img = np.zeros((2, 2, 4))
    img[:, :, :2] = 1.0
    img[:, :, 2:] = 5.0

    npt.assert_equal(connected_components(img)[1], 2)
    npt.assert_equal(connected_components(img, delta=3.9)[1], 2)
    npt.assert_equal(connected_components(img, delta=4.0)[1], 1)


def test_connected_components_no_foreground():
    labels, n = connected_components(np.zeros((3, 4, 5)))

    npt.assert_equal(n, 0)
    npt.assert_equal(labels, 0)


def test_connected_components_every_voxel_separate():
    # all intensities differ, so with delta=0 no two voxels join
    img = np.arange(1, 4 * 5 * 6 + 1, dtype='f8').reshape(4, 5, 6)

    labels, n = connected_components(img)

    npt.assert_equal(n, img.size)
    npt.assert_equal(len(np.unique(labels)), img.size)
