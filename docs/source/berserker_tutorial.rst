.. _berserker_tutorial:

==========================
Thetan Berserker Tutorial
==========================

The Thetan Berserker algorithm provides innovative ways to handle clustering tasks, outperforming standard sequential algorithms in certain scenarios.

This tutorial demonstrates how to use the Thetan Berserker (TB) and Thetan Sequential (TS) algorithms provided by the library.

First, we import the necessary components:

.. code-block:: python

    import numpy as np
    import pylab as pl
    from thetan import TS as ts
    from thetan import TB as tb

Next, let's create a utility to simulate some Gaussian cluster data:

.. code-block:: python

    def gaussian_clusters(n_local_samples, blob_dis=10, cluster_x=10, cluster_y=10):
        xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
        yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                               np.ravel(yy)[:, np.newaxis]))

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]

            local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        return X, y, gt_centroids

    def simulate_data(dist_type='gaussian', blob_dis=10, cluster_x=10, cluster_y=10, n_local_samples=500, scale=1):
        if dist_type == 'gaussian':
            X, y, gt_centroids = gaussian_clusters(n_local_samples=n_local_samples, blob_dis=blob_dis, cluster_x=cluster_x, cluster_y=cluster_y)

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        centroids = gt_centroids

        # Shuffle the points
        points_idx = np.arange(np.shape(points)[0])
        np.random.shuffle(points_idx)
        new_points = np.zeros(np.shape(points))
        new_labels = np.zeros(np.shape(labels))
        for i, idx in zip(range(np.shape(points)[0]), points_idx):
            new_points[i] = points[idx]
            new_labels[i] = labels[idx]

        return new_points, new_labels, gt_centroids

Now, let's generate the data:

.. code-block:: python

    theta_value = 3.6
    x = 20
    y = 10

    X, L, gt_centroids = simulate_data('gaussian', blob_dis=4.8, cluster_x=x, cluster_y=y)

    print('Number of features: ', X.shape[0])
    print('Number of GT centroids: ', gt_centroids.shape[0])

Running Thetan Sequential (TS)
------------------------------

We can run the sequential algorithm first to see how it performs:

.. code-block:: python

    print('\n >> Starting Thetan Sequential ...')
    ts_centroids, ts_labels = ts(X, theta_value, all_out=False)
    print('Done!')

Running Thetan Berserker (TB)
-----------------------------

Next, we run the Thetan Berserker algorithm:

.. code-block:: python

    print('\n >> Starting Thetan Berserker ...')
    tb_centroids, tb_labels = tb(X, theta_value)
    print('Done!')

Both functions return the found centroids and the corresponding labels for each data point.
