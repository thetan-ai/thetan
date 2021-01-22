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



X = disks_on_a_circle(n_samples=15, inner_radius=60, outer_radius=15, density=500)

print('Shape of X', X.shape)

from theta_machines_tests import theta_g2

centroids, labels = theta_g2(X, theta=20, num_shuffles=100)


from pylab import plot, show

plot(X[:, 0], X[:, 1], '.')
plot(centroids[:, 0], centroids[:, 1], '.')
show()

