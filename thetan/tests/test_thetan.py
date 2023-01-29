from time import time
import numpy as np
import numpy.testing as npt
from thetan.theta import reorder_labels, theta_g1, theta_means, reassign_uncertain_update_labels
import pylab as pl
from thetan.theta import distance_matrix as tdm
from thetan.berserker import berserker, candidate_labels


def plot_centroids_clusters(data, centroids, labels=None, title=None, exp_type=None, gt_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True):

    new_colors = np.zeros((len(data), 3))

    if centroids is not None:
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


def test_2d_data(verbose=False):

    n_local_samples = 400
    # n_local_samples = 1000

    dims = 2

    blob_dis = 5
    cluster_x = 40
    cluster_y = 10

    normal_distribution = True
    uniform_distribution = False
    compare_with_meanshift = True

    normal_sigma = 1.0
    uniform_low = -2.
    uniform_high = 2.

    theta_value = 3.6 # 3.0 still okay for uniform distributions with inf_norm

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


    # even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, metric='inf_norm', all_out=True, verbose=True)
    t0 = time()
    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, metric='euclidean', all_out=True, verbose=True)
    t1 = time()
    total_berserker_time = t1 - t0
    print('Total Thetan Berserker Time: ', total_berserker_time)

    if verbose:

        # plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=False)
        # plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
        # plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
        # plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
        # plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='Thetan Berserker ' + str(even_centroids_ex2.shape) + ' theta ' + str(theta_value), show=True)

    if compare_with_meanshift:

        from sklearn.cluster import MeanShift, estimate_bandwidth

        bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)

        for step in [3.0]: # np.arange(2.5, 3.5, step=0.25/2):
            t0 = time()
            ms = MeanShift(bandwidth=step, bin_seeding=True)
            ms.fit(points)
            t1 = time()


            meanshift_labels = ms.labels_
            meanshift_centers = ms.cluster_centers_
            print('Final shape:', meanshift_centers.shape[0])
            total_mean_shift_time = t1 - t0
            print('Total Mean Shift Time:', total_mean_shift_time)
            print('Speedup: ', total_mean_shift_time / total_berserker_time)

            plot_centroids_clusters(points, meanshift_centers, meanshift_labels, title='Meanshift ' + str(meanshift_centers.shape) + ' bandwidth ' + str(step), show=True)


    t = 1

    return even_centroids_ex2.shape[0]


def generate_circles(n_samples=15, inner_radius=60, outer_radius=15,
                     density=500, size=(200, 200), labellize=False,
                     return_coord=True):
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
    arr = np.zeros(size)
    label_value = 1
    coord_list = []
    for theta in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
        x = int(size[1]//2 + inner_radius * np.cos(theta))
        y = int(size[0]//2 + inner_radius * np.sin(theta))
        # arr[y, x] = 1
        if labellize:
            label_value += 1
        for _ in range(density):
            r = outer_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = x + r * np.cos(theta)
            y1 = y + r * np.sin(theta)
            if return_coord:
                coord_list.append([y1, x1])
            arr[int(y1), int(x1)] = label_value

    if return_coord:
        return arr, coord_list
    return arr


def test_grid_of_disks():

    # np.random.seed(0)
    sz = (50, 6)

    coords = []
    centroids = []

    for x, y in np.ndindex(sz):

        for d in range(100):
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = 5 * x + r * np.cos(theta)
            y1 = 5 * y + r * np.sin(theta)
            coords.append([x1, y1])

        centroids.append([5 * x, 5 * y])

    points = np.array(coords)
    print(points.shape[0])
    centroids = np.array(centroids, dtype='f8')
    print(centroids.shape[0])

    theta_value = 2.0

    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, all_out=True, verbose=True)

    print(even_centroids_ex2.shape[0])

    plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

    t = 1


def test_ring_of_disks():


    # Note: After outer_radius 13 the problem becomes much harder. An iterative version of berserker could resolve this issue

    _, points = generate_circles(n_samples=15, inner_radius=60, outer_radius=12,
                                 density=500, size=(200, 200), labellize=False,
                                 return_coord=True)
    points = np.asarray(points)

    print(points.shape)
    range_of_thr = np.arange(21, 23, 0.5)

    for theta_value in range_of_thr:

        print('theta value', theta_value)

        even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, all_out=True, verbose=True)

        print(even_centroids_ex2.shape[0])


        plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
        plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

        t = 1

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

    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, all_out=True, verbose=True)

    if verbose:
        plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
        plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

    npt.assert_almost_equal(even_centroids_ex2.shape[0], 2)


def candidate_labels(centroids, labels, initial_labels):
    """ Find labels for removal in cases of conflict
    """
    nice = np.ones(initial_labels.shape[0], dtype=initial_labels.dtype)
    updated_labels = initial_labels.copy()
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    for c in clusters:
        if len(c) > 1:
            cs = np.sort(c)
            candidates.extend(cs[1:])

    # caution here, deleting will change the labels of other Xs too which is not ideal

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l == can:
                nice[i] = 0
                updated_labels[i] = -1

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l > can:
                updated_labels[i] -= 1


    return nice, candidates, clusters, updated_labels


def test_candidate_labels():


    centroids = np.random.rand(5, 3)

    labels = np.array([0, 2, 2, 1, 3, 3], dtype=np.intp)

    initial_labels = np.array([0, 2, 5, 1, 3, 4], dtype=np.intp)

    nice, candidates, clusters, updated_labels = candidate_labels(centroids, labels, initial_labels)

    from thetan.theta import candidate_labels_c

    nice2, candidates2, clusters2, updated_labels2 = candidate_labels_c(centroids, labels, initial_labels)


    npt.assert_array_almost_equal(nice, nice2)
    npt.assert_array_almost_equal(candidates, candidates2)
    npt.assert_array_almost_equal(updated_labels, updated_labels2)

    t = 1


def test_3d_data(verbose=False):

    n_local_samples = 400
    # n_local_samples = 1000

    dims = 3

    blob_dis = 5 # 10
    cluster_x = 10 # 10
    cluster_y = 10 # 10
    cluster_z = 10 # 10

    normal_distribution = False
    uniform_distribution = True
    compare_with_kmeans = False
    compare_with_meanshift = True

    normal_sigma = 1.0
    uniform_low = -2.
    uniform_high = 2.

    theta_value = 3.6 # 3.8 looks ok too

    xx = np.arange(-1 * blob_dis*cluster_x // 2, blob_dis * cluster_x // 2, blob_dis)
    yy = np.arange(-1 * blob_dis*cluster_y // 2, blob_dis * cluster_y // 2, blob_dis)
    zz = np.arange(-1 * blob_dis*cluster_z // 2, blob_dis * cluster_z // 2, blob_dis)


    xx, yy, zz = np.meshgrid(xx, yy, zz)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis], np.ravel(zz)[:, np.newaxis]))

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

    from time import time


    print('-- Thetan Berserker --')

    t1 = time()
    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, metric='euclidean', all_out=True, verbose=True,
                                                                                                                                                     manage_outliers={'mode':'min_size', 'size_limit':200},
                                                                                                                                                     quality_over_speed=True)
    t2 = time()

    berserker_time = t2 - t1

    print('Berserker centroids: ', even_centroids_ex2.shape[0])
    print('Total berserker time ', berserker_time)

    unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2,
                                                     return_counts=True)

    thr = 0.8 # 0.4 okay too but a bit strict

    D = np.sqrt(tdm(even_centroids_ex2, centroids, metric='euclidean'))
    closest = np.min(D, axis=1)
    count = np.sum(closest < thr)



    print('berserker count ',  count)
    even_centroids_ex2_score = count / len(centroids)
    print('berserker score ',  even_centroids_ex2_score)

    if compare_with_kmeans:
        from sklearn.cluster import KMeans


        print('-- K-means++ --')
        t1 = time()
        # kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=1,
        #                 max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)

        kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=1,
                        max_iter=300, tol=0.0001, random_state=0).fit(points)
        t2 = time()

        kmeans_time = t2 - t1
        print('Total kmeans++ time ', kmeans_time)
        print('Berserker speedup ', kmeans_time/berserker_time)

        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_
        print('Kmeans++ centroids: ', kcentroids.shape[0])

        unique_labels_kmeans, unique_counts_kmeans = np.unique(klabels,
                                                               return_counts=True)

        D = np.sqrt(tdm(kcentroids, centroids, metric='euclidean'))
        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)

        print('kmeans++ count ', count)
        kscore = count / len(centroids)
        print('kmeans++ score ', kscore)


    if compare_with_meanshift:

        from sklearn.cluster import MeanShift, estimate_bandwidth

        # bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)


        print('-- Mean Shift --')

        for step in [3.0]: # np.arange(2.5, 3.5, step=0.25/2):
            t1 = time()
            ms = MeanShift(bandwidth=step, bin_seeding=True)
            ms.fit(points)
            t2 = time()
            meanshift_labels = ms.labels_
            meanshift_centroids = ms.cluster_centers_

            print('Meanshift centroids: ', meanshift_centroids.shape[0])

            meanshift_time = t2 - t1
        print('Total meanshift time ', meanshift_time)
        print('Berserker speedup ', meanshift_time/berserker_time)

        unique_labels_meanshift, unique_counts_meanshift = np.unique(meanshift_labels,
                                                                     return_counts=True)

        D = np.sqrt(tdm(meanshift_centroids, centroids, metric='euclidean'))
        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)

        print('Meanshift count ',  count)
        meanshift_score = count / len(centroids)
        print('Meanshift score ',  meanshift_score)

    if verbose:

        from fury import actor, window

        act = actor.dot(centroids, colors=(0.0, 0., 1.), opacity=1)
        act2 = actor.dot(points, colors=(0, 1.0, 0), opacity=0.1)

        # labels_of_interest = unique_labels_ex2[unique_counts_ex2 <= 60]

        # act3 = actor.dot(points[np.in1d(updated_labels_ex2, labels_of_interest)], colors=(1.0, 1., 1.0), opacity=.5)

        # act3 = actor.dot(p_centroids, colors=(1.0, 0.0, 0), opacity=1)
        act3 = actor.dot(even_centroids_ex2, colors=(1.0, 0.0, 0), opacity=1)
        # act3m = actor.dot(even_centroids_ex2[main_indices], colors=(1.0, 0.0, 0), opacity=1)
        # act3e = actor.dot(even_centroids_ex2[extra_indices], colors=(1.0, 1.0, 0), opacity=1)
        # act4 = actor.dot(kcentroids, colors=(1.0, 1.0, 0), opacity=1)
        act5 = actor.dot(meanshift_centroids, colors=(1.0, 0.5, 0), opacity=1)


        scene = window.Scene()
        scene.add(act)
        scene.add(act2)
        scene.add(act3)
        # scene.add(act3m)
        # scene.add(act3e)
        # scene.add(act4)
        scene.add(act5)
        show_manager = window.ShowManager(scene)
        show_manager.start()

        if compare_with_kmeans and compare_with_meanshift:

            from pylab import plot, show, grid, figure, title, legend
            # unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2, return_counts=True)
            figure()
            title('Thetan Berserker vs Kmeans++ vs MeanShift')
            plot(np.sort(unique_counts_ex2), '*-', label='Berserker')
            plot(np.sort(unique_counts_kmeans), '*-', label='Kmeans')
            plot(np.sort(unique_counts_meanshift), '*-', label='Meanshift')
            legend()
            grid()
            show()


        if compare_with_meanshift:

            from pylab import plot, show, grid, figure, title, legend
            # unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2, return_counts=True)
            figure()
            title('Thetan Berserker vs MeanShift')
            plot(np.sort(unique_counts_ex2), '*-', label='Berserker')
            plot(np.sort(unique_counts_meanshift), '*-', label='Meanshift')
            legend()
            grid()
            show()

    t = 1


def test_4d_data(verbose=False):

    n_local_samples = 400
    # n_local_samples = 1000

    dims = 4

    blob_dis = 5 # 10
    cluster_x = 10 # 10
    cluster_y = 10 # 10
    cluster_z = 10 # 10
    cluster_w = 10

    normal_distribution = False
    uniform_distribution = True
    compare_with_kmeans = False
    compare_with_meanshift = True

    normal_sigma = 1.0
    uniform_low = -2.
    uniform_high = 2.

    theta_value = 3.6 # 3.8 looks ok too

    xx = np.arange(-1 * blob_dis*cluster_x // 2, blob_dis * cluster_x // 2, blob_dis)
    yy = np.arange(-1 * blob_dis*cluster_y // 2, blob_dis * cluster_y // 2, blob_dis)
    zz = np.arange(-1 * blob_dis*cluster_z // 2, blob_dis * cluster_z // 2, blob_dis)
    ww = np.arange(-1 * blob_dis*cluster_w // 2, blob_dis * cluster_w // 2, blob_dis)

    xx, yy, zz, ww = np.meshgrid(xx, yy, zz, ww)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis], np.ravel(zz)[:, np.newaxis], np.ravel(ww)[:, np.newaxis]))

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

    print('>>> X.shape (points) ', X.shape)


    from time import time


    print('-- Thetan Berserker --')

    t1 = time()
    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker(points, theta_value, metric='euclidean', all_out=True, verbose=True,
                                                                                                                                                     manage_outliers={'mode':'min_size', 'size_limit':300},
                                                                                                                                                     quality_over_speed=True)
    t2 = time()

    berserker_time = t2 - t1

    print('Berserker centroids: ', even_centroids_ex2.shape[0])
    print('Total berserker time ', berserker_time)

    unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2,
                                                     return_counts=True)

    thr = 0.8 # 0.4 okay too but a bit strict

    D = np.sqrt(tdm(even_centroids_ex2, centroids, metric='euclidean'))
    closest = np.min(D, axis=1)
    count = np.sum(closest < thr)



    print('berserker count ',  count)
    even_centroids_ex2_score = count / len(centroids)
    print('berserker score ',  even_centroids_ex2_score)

    if compare_with_kmeans:
        from sklearn.cluster import KMeans


        print('-- K-means++ --')
        t1 = time()
        # kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=1,
        #                 max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)

        kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=1,
                        max_iter=300, tol=0.0001, random_state=0).fit(points)
        t2 = time()

        kmeans_time = t2 - t1
        print('Total kmeans++ time ', kmeans_time)
        print('Berserker speedup ', kmeans_time/berserker_time)

        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_
        print('Kmeans++ centroids: ', kcentroids.shape[0])

        unique_labels_kmeans, unique_counts_kmeans = np.unique(klabels,
                                                               return_counts=True)

        D = np.sqrt(tdm(kcentroids, centroids, metric='euclidean'))
        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)

        print('kmeans++ count ', count)
        kscore = count / len(centroids)
        print('kmeans++ score ', kscore)


    if compare_with_meanshift:

        from sklearn.cluster import MeanShift, estimate_bandwidth

        # bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)


        print('-- Mean Shift --')

        for step in [3.0]: # np.arange(2.5, 3.5, step=0.25/2):
            t1 = time()
            ms = MeanShift(bandwidth=step, bin_seeding=True)
            ms.fit(points)
            t2 = time()
            meanshift_labels = ms.labels_
            meanshift_centroids = ms.cluster_centers_

            print('Meanshift centroids: ', meanshift_centroids.shape[0])

            meanshift_time = t2 - t1
        print('Total meanshift time ', meanshift_time)
        print('Berserker speedup ', meanshift_time/berserker_time)

        unique_labels_meanshift, unique_counts_meanshift = np.unique(meanshift_labels,
                                                                     return_counts=True)

        D = np.sqrt(tdm(meanshift_centroids, centroids, metric='euclidean'))
        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)

        print('Meanshift count ',  count)
        meanshift_score = count / len(centroids)
        print('Meanshift score ',  meanshift_score)

    if verbose:

        from fury import actor, window

        # act = actor.dot(centroids, colors=(0.0, 0., 1.), opacity=1)
        # act2 = actor.dot(points, colors=(0, 1.0, 0), opacity=0.1)
        # act3 = actor.dot(even_centroids_ex2, colors=(1.0, 0.0, 0), opacity=1)
        # act5 = actor.dot(meanshift_centroids, colors=(1.0, 0.5, 0), opacity=1)


        # scene = window.Scene()
        # scene.add(act)
        # scene.add(act2)
        # scene.add(act3)
        # scene.add(act5)
        # show_manager = window.ShowManager(scene)
        # show_manager.start()

        if compare_with_kmeans and compare_with_meanshift:

            from pylab import plot, show, grid, figure, title, legend
            # unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2, return_counts=True)
            figure()
            title('Thetan Berserker vs Kmeans++ vs MeanShift')
            plot(np.sort(unique_counts_ex2), '*-', label='Berserker')
            plot(np.sort(unique_counts_kmeans), '*-', label='Kmeans')
            plot(np.sort(unique_counts_meanshift), '*-', label='Meanshift')
            legend()
            grid()
            show()


        if compare_with_meanshift:

            from pylab import plot, show, grid, figure, title, legend
            # unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2, return_counts=True)
            figure()
            title('Thetan Berserker vs MeanShift')
            plot(np.sort(unique_counts_ex2), '*-', label='Berserker')
            plot(np.sort(unique_counts_meanshift), '*-', label='Meanshift')
            legend()
            grid()
            show()

    t = 1


if __name__ == '__main__':

    ecs = []
    for i in range(100):
        # ecs.append(test_thetan_normal_uniform_grid(verbose=True))
        pass

    print(ecs)

    ecs = np.asarray(ecs)

    print(np.sum(ecs == 100))
    print(np.sum(ecs > 100))
    print(np.sum(ecs > 101))
    print(np.sum(ecs < 100))


    # test_grid_of_disks()
    # test_ring_of_disks()
    # test_two_triangles(True)
    # test_3d_data(verbose=True)
    test_4d_data(verbose=True)
    # test_2d_data(verbose=True)


    # test_candidate_labels()

    # test_reorder_labels()
    # test_build_nice()
    t = 2
    pass
