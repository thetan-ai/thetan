import numpy as np
from pylab import plot, show, scatter, figure, title, savefig, axis


def disks_on_a_circle(n_samples=15, inner_radius=60, outer_radius=15,
                     density=500):
    """Generate small circle on a bigger one.

    Parameters
    ----------
    n_samples: int
        number of outer circle

    inner_radius: int
        radius of the bigger circle

    outer_radius: int
        radius of the small circle

    density: int
        number of point on each circle uniformly distributed

    size: tuple
        2d image size

    labellize: bool
        Switch on/off if you want to labellize each circle or not

    Returns
    -------
    arr: 2d array
        numpy array containing the drawing

    """
    label_value = 1
    coord_list = []
    for theta in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
        x = inner_radius * np.cos(theta)
        y = inner_radius * np.sin(theta)
        for _ in range(density):
            r = outer_radius * np.sqrt(np.random.rand())
            # r = outer_radius + 0.01*np.random.normal() # creates rings
            theta = np.random.random() * 2 * np.pi
            x1 = x + r * np.cos(theta)
            y1 = y + r * np.sin(theta)
            coord_list.append([x1, y1])

    return np.array(coord_list)


def show_clusters(method, centroids, labels):
    figure()
    title('Method is ' + method)
    unique_colors = np.random.rand(centroids.shape[0], 4)
    colors = np.zeros((labels.shape[0], 4), dtype='f8')
    for i in range(X.shape[0]):
        colors[i] = unique_colors[labels[i]]
        colors[:, 3] = 1
    axis('equal')
    scatter(X[:, 0], X[:, 1], c=colors)
    # scatter(scentroids[:, 0], scentroids[:, 1], c=np.array([0.6, .6, .6]))
    scatter(centroids[:, 0], centroids[:, 1])
    #savefig(method + '.png')


if __name__ == "__main__":


    from robotikos.theta import theta_g1, assign
    from theta_machines_tests import theta_g2, theta_g2x, labels_to_centroids
    from robotikos.fgm import fgn_shuffles

    from robotikos.metric import IdentityFeature
    from robotikos.metric import NormalizedSumSquaredDiffMetric

    from sklearn.cluster import KMeans

    methods = ['t_g2', 'fgn_shuffles', 'kmeans++']
    # methods = ['fgn_shuffles', 'kmeans++']
    # for theta in [15., 16., 18., 19., 20., 21.]:

    # notice that n_disks=15 challenging for theta_g2

    n_disks = 15
    X = disks_on_a_circle(n_samples=n_disks, inner_radius=60, outer_radius=15, density=500)

    # n_disks = 15
    # X2 = disks_on_a_circle(n_samples=n_disks, inner_radius=120, outer_radius=15, density=500)

    # n_disks = 15 * 2

    # X = np.concatenate((X, X2))

    # n_disks = 12
    # X = disks_on_a_circle(n_samples=n_disks, inner_radius=60, outer_radius=15, density=500)

    # n_disks = 120
    # X = disks_on_a_circle(n_samples=n_disks, inner_radius=600, outer_radius=15, density=500)

    print('Shape of X', X.shape)


    for theta in [20.]:

        if 't_g2' in methods:
            centroids, labels = theta_g1(X, theta=theta)
            show_clusters('t_g1', centroids, labels)

            # ncentroids, nlabels = theta_g2(centroids, theta=theta, num_shuffles=50)
            # labels = assign(X, ncentroids)
            # centroids = labels_to_centroids(X, labels)
            # show_clusters('t_g2X', centroids, labels)

            centroids, labels = theta_g2(X, theta=theta, num_shuffles=50)
            show_clusters('t_g2', centroids, labels)

            centroids, labels, interm_centroids = theta_g2x(X, theta=theta, num_partitions=10, num_centroid_shuffles=10)
            show_clusters('t_g2x', centroids, labels)
            show_clusters('t_g2x_interm', interm_centroids, labels)


        if 'fgn_shuffles' in methods:

            metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
            all_levels, s_centroids = fgn_shuffles(X[:, None, :], thresholds=[theta * theta / X.shape[1]],
                                                   metric=metric,
                                                   num_shuffles=50, mode='fgn',
                                                   verbose=True, ret_shuffles_centroids=True)
            centroids_tmp = all_levels[1]['centroids']
            centroids = np.zeros((len(centroids_tmp), 2))
            for ci, c in enumerate(centroids_tmp.values()):
                centroids[ci] = np.squeeze(c)
            labels = all_levels[1]['labels']

            scentroids_tmp = s_centroids[1]
            scentroids = np.zeros((len(scentroids_tmp), 2))
            for ci, c in enumerate(scentroids_tmp):
                scentroids[ci] = np.squeeze(c)

            show_clusters('fgn_shuffles', centroids, labels)

        if 'kmeans++' in methods:

            kmeans = KMeans(n_clusters=n_disks, init='k-means++', n_init=10,
                            max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(X)

            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

            show_clusters('kmeans++', centroids, labels)


    show()

