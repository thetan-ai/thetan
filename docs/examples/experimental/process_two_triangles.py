import os
import numpy as np
import numpy.testing as npt
from thetan.berserker import berserker, berserker2, tsg_randomized
from thetan.theta import theta_g1
from thetan.viz import plot_centroids_clusters


def points_on_triangle(v, n):
    """ Give n random points uniformly on a triangle.

    The vertices of the triangle are given by the shape
    (2, 3) array *v*: one vertex per row.
    """
    x = np.sort(np.random.rand(2, n), axis=0)

    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v


def two_triangles(inbetween_distance = 0.3, theta_value = 0.6):

    v = np.array([(-0.5, 0.5), (-0.5, -0.5), (0.5, 0)])

    points = points_on_triangle(v, 2000)

    v = np.array([(inbetween_distance + 0.5, 0.5), (inbetween_distance + 0.5, -0.5), (inbetween_distance - 0.5, 0)])

    points2 = points_on_triangle(v, 2000)

    points = np.concatenate((points, points2), axis=0)


    # even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, all_out=True, verbose=True)

    # if verbose:
    #     plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=True)
    #     plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
    #     plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
    #     plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
    #     plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
    #     plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

    # npt.assert_almost_equal(even_centroids_ex2.shape[0], 2)

    # TODO: compare with TSG, TB (berserker2), TB (one)

    for run_no in range(20):

        dname = 'C:/Users/Eleftherios/Data/two_triangles_experiments/20_theta_' + str(inbetween_distance) + '_' + str(theta_value) + '_run_' + str(run_no) + '/'
        os.mkdir(dname)

        # Thetan Sequential
        ts_centroids, ts_labels = theta_g1(points, theta_value, metric='euclidean', all_out=False)
        plot_centroids_clusters(points, ts_centroids, ts_labels, title='TS ' + str(ts_centroids.shape), show=False, maxwindow=False, savefn=dname + 'ts_two_triangles.png', centroids_label=False)

        # Thetan Berserker v1
        tb1_centroids, tb1_labels = berserker(points, theta_value, metric='euclidean', all_out=False)
        plot_centroids_clusters(points, tb1_centroids, tb1_labels, title='TB v1' + str(tb1_centroids.shape), show=False, maxwindow=False, savefn=dname + 'tb_v1_two_triangles.png', centroids_label=False)

        # Thetan Berserker v2
        tb_centroids, tb_labels = berserker2(points, theta_value, metric='euclidean', all_out=False)
        plot_centroids_clusters(points, tb_centroids, tb_labels, title='TB v2' + str(tb_centroids.shape), show=False, maxwindow=False, savefn=dname + 'tb_v2_two_triangles.png', centroids_label=False)

        # Thetan Sequential Randomized
        tsr_centroids, tsr_labels, sum_of_time = tsg_randomized(points, theta_value, metric='euclidean', num_shuffles=10)
        plot_centroids_clusters(points, tsr_centroids, tsr_labels, title='TSR  ' + str(tsr_centroids.shape), show=False, maxwindow=False, savefn=dname + 'tsr_two_triangles.png', centroids_label=False)


if __name__ == '__main__':

    # two_triangles(inbetween_distance=1.2, theta_value=0.6)
    # two_triangles(inbetween_distance=0.9, theta_value=0.6)
    # two_triangles(inbetween_distance=0.6, theta_value=0.6)
    # two_triangles(inbetween_distance=0.3, theta_value=0.6)
    # two_triangles(inbetween_distance=0.9, theta_value=0.6)
    # two_triangles(inbetween_distance=0.9, theta_value=0.7)
    # two_triangles(inbetween_distance=1.0, theta_value=0.6)
    # two_triangles(inbetween_distance=1.1, theta_value=0.7)
    # two_triangles(inbetween_distance=1.0, theta_value=0.5)
    two_triangles(inbetween_distance=1.0, theta_value=0.65)
