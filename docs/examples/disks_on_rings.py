import numpy as np

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
            r = outer_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = x + r * np.cos(theta)
            y1 = y + r * np.sin(theta)
            coord_list.append([x1, y1])

    return np.array(coord_list)


if __name__ == "__main__":


    X = disks_on_a_circle(n_samples=15, inner_radius=60, outer_radius=15, density=500)

    print('Shape of X', X.shape)

    from pylab import plot, show, scatter, figure, title

    from theta_machines_tests import theta_g2
    from robotikos.fgm import fgn_shuffles

    from robotikos.metric import IdentityFeature
    from robotikos.metric import NormalizedSumSquaredDiffMetric



    # for theta in [15., 16., 18., 19., 20., 21.]:
    for theta in [20.]:

        metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
        # centroids, labels = theta_g2(X, theta=theta, num_shuffles=50)

        all_levels = fgn_shuffles(X[:, None, :], thresholds=[theta*theta/X.shape[1]],
                                  metric=metric,
                                  num_shuffles=50, mode='fgn_ma', verbose=True)
        centroids_tmp = all_levels[1]['centroids']
        centroids = np.zeros((len(centroids_tmp), 2))
        for ci, c in enumerate(centroids_tmp.values()):
            centroids[ci] = c
        labels = all_levels[1]['labels']

        unique_colors = np.random.rand(centroids.shape[0], 4)
        colors = np.zeros((labels.shape[0], 4), dtype='f8')
        for i in range(X.shape[0]):
            colors[i] = unique_colors[labels[i]]
        colors[:, 3] = 1

        figure()
        title('Theta is ' + str(theta))
        scatter(X[:, 0], X[:, 1], c=colors[:, ])
        scatter(centroids[:, 0], centroids[:, 1])

    show()

