import numpy as np
import numpy.testing as npt
from thetan.theta import reorder_labels, candidate_labels, theta_g1, theta_means, reassign_uncertain_update_labels
from thetan.berseker import berserker
import pylab as pl


def plot_centroids_clusters(data, centroids, labels=None, title=None, exp_type=None, gt_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)


    if labels is not None:

        for ind in range(len(data)):
            new_colors[ind] = cluster_colors[labels[ind]]

    fig = pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)

    if centroids is not None:

        pl.plot(centroids[:, 0], centroids[:, 1], 'rx')

        for c in range(centroids.shape[0]):
            pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if gt_centroids is not None:
        pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'bx')

    if uncertain_data is not None:
        pl.plot(uncertain_data[:, 0], uncertain_data[:, 1], 'g.')

    cnt = 0
    if final_counts is not None:
        for i in range(final_counts.shape[0]):
            prev = cnt
            cnt += final_counts[i]
            if final_counts[i] > 1:
                pl.text(data[i, 0], data[i, 1], str(final_nearc_value[prev: cnt]) + ' ' + str(labels[i]), fontsize=8)

    if merged_centroids is not None:
        pl.plot(merged_centroids[:, 0], merged_centroids[:, 1], 'mo')

    if show:
        pl.show()



def build_nice2(centroids, labels):
    nice = np.ones(labels.shape[0], dtype=labels.dtype)
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    # clusters = {}
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    for c in clusters:
        if len(c) > 1:
            maxc = np.max(c)
            candidates.append(maxc)

    # deleting will change the labels of other Xs too which is not ideal
    # centroids_reduced = np.delete(final_centroids, candidates, axis=0)

    for i, f in enumerate(labels):
        for can in candidates:
            if labels[i] == can:
                nice[i] = 0

    return nice, candidates, clusters


def test_build_nice():

    centroids = np.array([[0, 0, 0],
                          [0.2, 0, 0],
                          [3, 3, 3.],
                          [4, 4, 4.],
                          [5, 5, 5.],
                          [6, 6, 6],
                          [6.2, 6, 6]])

    labels = np.array([0, 0, 1, 2, 3, 4, 4, 4])

    nice, candidates, clusters = build_nice2(centroids, labels)


    t = 1


def test_reorder_labels():

    centroids = np.random.rand(7, 3)
    labels = np.array([0, 0, 1, 2, 4, 5, 6, 6])
    labels_updated = reorder_labels(centroids,
                                    labels,
                                    None, [2, 5])

    npt.assert_equal(labels_updated,
                     np.array([0, 0, 1, 2, 3, 4, 4, 4]))


def test_thetan_oic():

    n_local_samples = 400
    # n_local_samples = 1000

    dims = 2

    blob_dis = 5
    cluster_x = 10
    cluster_y = 10

    normal_distribution = True
    uniform_distribution = False

    normal_sigma = 1.0
    uniform_low = -2.
    uniform_high = 2.

    theta_value = 3.6

    xx = np.arange(-1 * blob_dis*cluster_x // 2, blob_dis * cluster_x // 2, blob_dis)
    yy = np.arange(-1 * blob_dis*cluster_y // 2, blob_dis * cluster_y // 2, blob_dis)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], dims))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)

    for i in range(n_centres.shape[0]):
        cpos = n_centres[i]

        if normal_distribution:
            local_X = cpos + np.random.normal(0, normal_sigma, (n_local_samples, dims))

        if uniform_distribution:
            local_X = cpos + np.random.uniform(uniform_low,
                                               uniform_high,
                                               (n_local_samples, dims))

        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    gt_centroids = np.array(gt_centroids)

    points = X
    labels = y
    sz = len(n_centres)

    centroids = gt_centroids


    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, all_out=True, verbose=True)

    verbose = True

    if verbose:

        plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
        plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)


    t = 1

    return res[0].shape[0]


if __name__ == '__main__':

    ecs = []
    for i in range(100):
        ecs.append(test_thetan_oic())

    print(ecs)

    ecs = np.asarray(ecs)

    print(np.sum(ecs == 100))
    print(np.sum(ecs > 100))
    print(np.sum(ecs > 101))
    print(np.sum(ecs < 100))


    # test_reorder_labels()
    # test_build_nice()
    t = 2
    pass
