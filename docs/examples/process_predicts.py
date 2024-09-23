from time import time
import numpy as np
from thetan.berserker import berserker2, berserker2_clean, random_walk_auto
import pylab as pl
import tracemalloc


def gaussian_clusters(n_local_samples, blob_dis = 10, cluster_x = 10, cluster_y = 10):

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


def uniform_clusters(scale = 4, n_local_samples = 30, blob_dis = 33, cluster_per_dim = 10):

    xx = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)
    yy = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                        np.ravel(yy)[:, np.newaxis]))

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], 2))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
    for i in range(n_centres.shape[0]):
        cpos = n_centres[i]
        local_X = cpos + scale*(np.random.rand(n_local_samples, 2) - 0.5)

        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    return X, y, gt_centroids


def disk_clusters(sz = (20, 6), scale=5, samples=100):

    coords = []
    centroids = []
    Y = [] # labels

    cnt = 0
    for x, y in np.ndindex(sz):

        for d in range(samples):
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = scale * x + r * np.cos(theta)
            y1 = scale * y + r * np.sin(theta)
            coords.append([x1, y1])
            Y.append(cnt)

        cnt += 1
        centroids.append([scale * x, scale * y])

    points = np.array(coords)
    centroids = np.array(centroids, dtype='f4')

    return points, Y, centroids


def plot_centroids_clusters(data, centroids, labels, title, exp_type, initial_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True, dpath=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(data.shape[0]):
        new_colors[ind] = cluster_colors[labels[ind]]

    fig = pl.figure(figsize=(30, 15))
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')

    # for c in range(centroids.shape[0]):
    #    pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')

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
    if dpath is not None:
        pl.savefig(dpath + '2_' + exp_type + '_' + title + '.png', dpi=300)
        pl.close(fig)


def simulate_data(dist_type='gaussian',blob_dis=10, cluster_x=10, cluster_y=10):

    print('Simulating data')

    if dist_type == 'gaussian':
        X, y, gt_centroids = gaussian_clusters(n_local_samples=500, blob_dis = blob_dis, cluster_x = cluster_x, cluster_y = cluster_y) # cluster_x should be 30

    if dist_type == 'uniform':
        X, y, gt_centroids = uniform_clusters(scale = 4, n_local_samples = 30, blob_dis = 33, cluster_per_dim = 10)

    if dist_type == 'disk':
        X, y, gt_centroids = disk_clusters(sz = (20, 6), scale=5, samples=100)

    gt_centroids = np.array(gt_centroids)

    points = X
    labels = y
    # sz = len(n_centres)

    centroids = gt_centroids

    print('Before shuffling')
    print(points[:10])

    points_idx = np.arange(np.shape(points)[0])
    np.random.shuffle(points_idx)
    new_points = np.zeros(np.shape(points))
    new_labels = np.zeros(np.shape(labels))
    for i, idx in zip(range(np.shape(points)[0]), points_idx):
        new_points[i] = points[idx]
        new_labels[i] = labels[idx]

    points = new_points
    labels = new_labels

    return points, labels, gt_centroids


if __name__ == '__main__':

    X, L, gt_centroids = simulate_data('gaussian', blob_dis=20, cluster_x=30, cluster_y=10)

    print('Number of features ', X.shape[0])
    print('Number of GT centroids ', gt_centroids.shape[0])
    print('Size in Bytes', X.nbytes // 1024 ** 2)

    verbose = True

    print('\n >> Aiming to predict theta in linear time ...')

    dists = random_walk_auto(X, reps=10)
    dists = np.asarray(dists)

    pl.figure()
    # pl.plot(dists, '*', alpha=0.2)

    pl.hist(dists.ravel(), bins=50)
    pl.show()

    theta_value = 5.0

    print('\n >> Starting Berserker 2 (clean version)...')

    t = time()
    tracemalloc.start()
    centroids, labels = berserker2_clean(X, theta_value)
    amem, peakmem = tracemalloc.get_traced_memory()
    peakmem = peakmem // (1024**2)
    tracemalloc.stop()
    duration = time() - t
    print('Duration is ', duration)
    print('Peak memory ', peakmem)

    if verbose:
        plot_centroids_clusters(X, centroids, labels, 'TB_cln_', exp_type='_runtime_', initial_centroids=gt_centroids, show=True, dpath=None)

    
