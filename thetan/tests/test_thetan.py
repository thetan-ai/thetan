from time import time
import numpy as np
import numpy.testing as npt
from thetan import berserker, sequential
from thetan import centroids_from_labels
from thetan.viz import plot_centroids_clusters


def points_on_triangle(v, n):
    """ Give n random points uniformly on a triangle.

    The vertices of the triangle are given by the shape
    (2, 3) array *v*: one vertex per row.
    """
    x = np.sort(np.random.rand(2, n), axis=0)

    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v


def test_two_triangles(verbose=False):
    v = np.array([(-0.5, 0.5), (-0.5, -0.5), (0.5, 0)])
    points = points_on_triangle(v, 2000)

    inbetween_distance = 0.3

    v = np.array([(inbetween_distance + 0.5, 0.5), (inbetween_distance + 0.5, -0.5), (inbetween_distance - 0.5, 0)])
    points2 = points_on_triangle(v, 2000)

    points = np.concatenate((points, points2), axis=0)

    theta_value = .6

    even_centroids_ex2, updated_labels_ex2 = berserker(points, theta_value)

    if verbose:
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

    npt.assert_almost_equal(even_centroids_ex2.shape[0], 2)

def _centroids_from_labels(points, labels):

    new_centroids = []
    for i, l in enumerate(range(labels.max() + 1)):
        # print(i)
        avg = np.average(points[labels==l], axis=0)
        new_centroids.append(avg)

    t = 0
    return np.ascontiguousarray(new_centroids)

def test_centroids_from_labels():

    dims = 3
    theta_value = 50

    points = 1000 * (np.random.rand(1000, dims) - 0.5)

    tb1 = time()
    centroids1, labels1 = sequential(points, theta_value)
    tb2 = time()


    t0 = time()
    _centroids = _centroids_from_labels(points, labels1)
    t1 = time()
    centroids2 = centroids_from_labels(points, labels1)
    t2 = time()

    print('G1 version')
    print(centroids1.shape)
    print(centroids1[:4])

    print('Pythonic version', t1 - t0)
    print(_centroids[:4])
    print(_centroids.shape)

    print('Cythonic version', t2 - t1)
    print(centroids2[:4])
    print(centroids2.shape)

    # plot_centroids_clusters(points, _centroids, labels1, title='Yo before', show=False)
    # plot_centroids_clusters(points, centroids2, labels1, title='Yo after', show=True)

    npt.assert_array_almost_equal(centroids2, centroids1)
    npt.assert_array_almost_equal(centroids2, _centroids)

if __name__ == '__main__':
    test_two_triangles()
    test_centroids_from_labels()
    print('Tests passed!')
