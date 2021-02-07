from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.datasets import make_blobs
import pylab as pl
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from robotikos.theta import theta_g1, assign
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
from robotikos.metricspeed import distance_matrix

def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)
    
    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids


def theta_g2(X, theta, metric='euclidean', num_shuffles=10):
    sum_of_time = 0
    all_centroids = {}
    X_work = X.copy()

    num_centroids = []
    timings = []
    for s in range(num_shuffles):

        np.random.shuffle(X_work)
        X_work_cp = np.ascontiguousarray(X_work)
        t1 = time()
        centroids, _ = theta_g1(X_work_cp, theta=theta, metric=metric)
        t2 = time()        
        timings.append(t2-t1)

        all_centroids[s] = centroids
        num_centroids.append(len(centroids))

    print('Time for tg1 calls', np.sum(timings))
    sum_of_time += np.sum(timings)
    total_num_centroids = np.sum(num_centroids)
    X_centroids = np.zeros((total_num_centroids, X_work.shape[1]), dtype='f8')

    offset = 0

    for s in range(num_shuffles):

        X_centroids[offset:offset + all_centroids[s].shape[0]] = all_centroids[s]
        offset += all_centroids[s].shape[0]
    
    t1 = time()
    new_centroids, _ = theta_g1(X_centroids, theta=theta, metric=metric)
    t2 = time()
    sum_of_time += t2 - t1
    print('Time for tg1 on centroids', t2 - t1)
    

    t1 = time()
    final_indices = assign(X, new_centroids, metric=metric)
    t2 = time()
    sum_of_time += t2 - t1
    print('Time for assign calls', t2 - t1)
    final_centroids = labels_to_centroids(X, final_indices)

    return final_centroids, final_indices, sum_of_time

def evaluate_centroids(ecentroids=None, centroids=None, thr=0.5, mode='NMI', X=None, label=None, true_label=None):
    if mode == 'SSD':
        metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
        D = np.sqrt(distance_matrix(metric,
                                    np.ascontiguousarray(ecentroids[:, None, :]),
                                    np.ascontiguousarray(centroids[:, None, :])))

        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)
        score = count / len(centroids)

    elif mode == 'NMI':
        score = normalized_mutual_info_score(true_label, label)

    elif mode == 'SIL':
        score = silhouette_score(X, label)
    
    elif mode == 'FMS':
        score = fowlkes_mallows_score(true_label, label)


    else:
        print('no mode')
        score = 0
    
    
    
    print('>>>>> ', score)
    return score




def plot_centroids_clusters(data, centroids, labels, title, exp_type, initial_centroids=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(len(points)):
        new_colors[ind] = cluster_colors[labels[ind]]

    fig = pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')
    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')
    pl.savefig(exp_type+'_'+title+'.png', dpi=300)
    pl.close(fig)


#exp_type = 'easy'
exp_type = 'hard'
if exp_type == 'easy':
    methods = ['kmeans', 'kmeans++', 'tm', 'mb_kmeans', 'birch']
    #methods = ['birch']
    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    evaluation_thr = 0.1
    verbose = True
    show_thr = False
    radius = 4
    n_local_samples = 30
    blob_dis = 33
    cluster_per_dim = 10
    scores = {}
    timings = {}

    if exp_type == 'blobs':
        # Generate centers for the blobs so that it forms a 10 X 10 grid.
        # xx = np.linspace(-22, 22, 10)
        # yy = np.linspace(-22, 22, 10)
        
        xx = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)
        yy = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                            np.ravel(yy)[:, np.newaxis]))

        #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
        # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]
            local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
            #local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)
        #centroids = np.array(n_centres, dtype='f4')
        centroids = gt_centroids

    if exp_type == 'easy':
        # Generate centers for the blobs so that it forms a 10 X 10 grid.
        # xx = np.linspace(-22, 22, 10)
        # yy = np.linspace(-22, 22, 10)
        blob_dis = 10
        cluster_x = 10
        cluster_y = 10
        xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
        yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                            np.ravel(yy)[:, np.newaxis]))

        #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
        # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]
            #local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
            local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)
        #centroids = np.array(n_centres, dtype='f4')
        centroids = gt_centroids

    if exp_type == 'hard':
        blob_dis = 5
        cluster_x = 10
        cluster_y = 10
        xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
        yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                            np.ravel(yy)[:, np.newaxis]))

        #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
        # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]
            #local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
            local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)
        #centroids = np.array(n_centres, dtype='f4')
        centroids = gt_centroids

    if exp_type == 'disks':

        np.random.seed(0)
        sz = (50, 6)

        coords = []
        centroids = []

        for x, y in np.ndindex(sz):

            for d in range(100):
                r = np.sqrt(np.random.random())
                theta = np.random.random() * 2 * np.pi
                x1 = 5 * x + r * np.cos(theta)
                # y1 = 5 * y + 2 * r * np.sin(theta)
                y1 = 5 * y + r * np.sin(theta)
                coords.append([x1, y1])
                
            centroids.append([5 * x, 5 * y])

        points = np.array(coords)
        centroids = np.array(centroids, dtype='f4')
    """
    plot_centroids_clusters(points, centroids, labels,
                                    'original',
                                    initial_centroids=centroids)
    """


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

    print('After shuffling')
    print(points[:10])

    print('Number of points ', points.shape)

    print('Number of gold centroids ', np.prod(sz))

    if 'kmeans' in methods:

        t1 = time()
        kmeans = KMeans(n_clusters=np.prod(sz), init='random', n_init=1,
                        max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)
        t2 = time()

        kmeans_time = t2 - t1
        print('kmeans ', kmeans_time)

        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_
        if verbose:
            plot_centroids_clusters(points, kcentroids, klabels,
                                    'kmeans', exp_type,
                                    initial_centroids=centroids)
        scores['kmeans'] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['kmeans'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['kmeans'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels)
            elif mode == 'SSD':
                scores['kmeans'][mode] = evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode)
        timings['kmeans'] = kmeans_time

        tmp = 0


    if 'kmeans++' in methods:

        t1 = time()
        kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=1,
                        max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)
        t2 = time()

        kmeans_time = t2 - t1
        print('kmeans++ ', kmeans_time)

        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_
        if verbose:
            plot_centroids_clusters(points, kcentroids, klabels,
                                    'kmeans++', exp_type,
                                    initial_centroids=centroids)

        scores['kmeans++'] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['kmeans++'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['kmeans++'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels)
            elif mode == 'SSD':
                scores['kmeans++'][mode] = evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode)
        timings['kmeans++'] = kmeans_time

        tmp = 0

    if 'tm' in methods:
        theta = 6.0

        t1 = time()
        tcentroids, tlabels = theta_g1(points, theta=theta,)

        t2 = time()

        tm_time = t2 - t1
        print('tm ', tm_time)
        print('Speedup', kmeans_time/tm_time)

        if verbose:
            plot_centroids_clusters(points, tcentroids, tlabels, 'tm', exp_type, initial_centroids=centroids)
        timings['tm'] = tm_time


    if 'birch' in methods:

        for bt in [2.7]:  # np.linspace(1.5, 2.5, 10):

            t1 = time()
            # birch = Birch(threshold=1.7, n_clusters=None).fit(points)
            # birch = Birch(threshold=1.7, n_clusters=int(np.prod(sz))).fit(points)
            # birch = Birch(threshold=3.0, n_clusters=int(np.prod(sz))).fit(points)
            birch = Birch(threshold=bt, n_clusters=int(np.prod(sz))).fit(points)

            t2 = time()

            birch_time = t2 - t1
            print('birch_time ', birch_time)

            birch_labels = birch.labels_
            birch_centroids = birch.subcluster_centers_
            print('# centroids ', len(birch_centroids))

            ext = str(np.round(bt, 2))
            if not show_thr:
                ext = ''

            if verbose:
                plot_centroids_clusters(points, birch_centroids, birch_labels, 'birch ' + ext, exp_type, initial_centroids=centroids)

            scores['birch ' + ext] = {}
            for mode in modes:
                if mode == 'SIL':
                    scores['birch ' + ext][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=birch_labels)
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['birch ' + ext][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=birch_labels, true_label=labels)
                elif mode == 'SSD':
                    scores['birch ' + ext][mode] = evaluate_centroids(birch_centroids, centroids, evaluation_thr, mode=mode)
            timings['birch ' + ext] = birch_time

    if 'mb_kmeans' in methods:

        t1 = time()
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=int(np.prod(sz)),
                            batch_size=100,
                            n_init=1, max_no_improvement=10, verbose=0,
                            random_state=0).fit(points)
        t2 = time()
        mb_kmeans_time = t2 - t1
        print('mb_kmeans_time ', mb_kmeans_time)

        mbk_labels = mbk.labels_
        mbk_centroids = mbk.cluster_centers_
        print('# centroids ', len(mbk_centroids))
        if verbose:
            plot_centroids_clusters(points, mbk_centroids, mbk_labels, 'mini_batch', exp_type, initial_centroids=centroids)

        scores['mini_batch'] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['mini_batch'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=mbk_labels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['mini_batch'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=mbk_labels, true_label=labels)
            elif mode == 'SSD':
                scores['mini_batch'][mode] = evaluate_centroids(mbk_centroids, centroids, evaluation_thr, mode=mode)
        timings['mini_batch'] = mb_kmeans_time


    fig = pl.figure()
    pl.title('Timings')
    x_tmp = np.arange(len(timings.values()))
    pl.bar(x_tmp, timings.values(), color=(1, 0, 0., 0.7))
    pl.xticks(x_tmp, timings.keys())
    pl.savefig('easy_Timing.png', dpi=300)
    pl.close(fig)
    print('easy timings', timings)

    t = 1
exp_type = 'hard'
if exp_type == 'hard':
    methods = ['kmeans', 'kmeans++', 'tm', 'mb_kmeans', 'birch']
    #methods = ['birch']
    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    evaluation_thr = 0.1
    verbose = True
    show_thr = False
    radius = 4
    n_local_samples = 30
    blob_dis = 33
    cluster_per_dim = 10
    scores = {}
    timings = {}

    if exp_type == 'blobs':
        # Generate centers for the blobs so that it forms a 10 X 10 grid.
        # xx = np.linspace(-22, 22, 10)
        # yy = np.linspace(-22, 22, 10)
        
        xx = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)
        yy = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                            np.ravel(yy)[:, np.newaxis]))

        #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
        # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]
            local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
            #local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)
        #centroids = np.array(n_centres, dtype='f4')
        centroids = gt_centroids

    if exp_type == 'easy':
        # Generate centers for the blobs so that it forms a 10 X 10 grid.
        # xx = np.linspace(-22, 22, 10)
        # yy = np.linspace(-22, 22, 10)
        blob_dis = 10
        cluster_x = 10
        cluster_y = 10
        xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
        yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                            np.ravel(yy)[:, np.newaxis]))

        #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
        # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]
            #local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
            local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)
        #centroids = np.array(n_centres, dtype='f4')
        centroids = gt_centroids

    if exp_type == 'hard':
        blob_dis = 5
        cluster_x = 10
        cluster_y = 10
        xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
        yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

        xx, yy = np.meshgrid(xx, yy)
        n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                            np.ravel(yy)[:, np.newaxis]))

        #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
        # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]
            #local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
            local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)
        #centroids = np.array(n_centres, dtype='f4')
        centroids = gt_centroids

    if exp_type == 'disks':

        np.random.seed(0)
        sz = (50, 6)

        coords = []
        centroids = []

        for x, y in np.ndindex(sz):

            for d in range(100):
                r = np.sqrt(np.random.random())
                theta = np.random.random() * 2 * np.pi
                x1 = 5 * x + r * np.cos(theta)
                # y1 = 5 * y + 2 * r * np.sin(theta)
                y1 = 5 * y + r * np.sin(theta)
                coords.append([x1, y1])
                
            centroids.append([5 * x, 5 * y])

        points = np.array(coords)
        centroids = np.array(centroids, dtype='f4')
    """
    plot_centroids_clusters(points, centroids, labels,
                                    'original',
                                    initial_centroids=centroids)
    """


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

    print('After shuffling')
    print(points[:10])

    print('Number of points ', points.shape)

    print('Number of gold centroids ', np.prod(sz))

    if 'kmeans' in methods:

        t1 = time()
        kmeans = KMeans(n_clusters=np.prod(sz), init='random', n_init=50,
                        max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)
        t2 = time()

        kmeans_time = t2 - t1
        print('kmeans ', kmeans_time)

        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_
        if verbose:
            plot_centroids_clusters(points, kcentroids, klabels,
                                    'kmeans', exp_type,
                                    initial_centroids=centroids)
        scores['kmeans'] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['kmeans'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['kmeans'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels)
            elif mode == 'SSD':
                scores['kmeans'][mode] = evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode)
        timings['kmeans'] = kmeans_time

        tmp = 0


    if 'kmeans++' in methods:

        t1 = time()
        kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=50,
                        max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)
        t2 = time()

        kmeans_time = t2 - t1
        print('kmeans++ ', kmeans_time)

        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_
        if verbose:
            plot_centroids_clusters(points, kcentroids, klabels,
                                    'kmeans++', exp_type,
                                    initial_centroids=centroids)

        scores['kmeans++'] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['kmeans++'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['kmeans++'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels)
            elif mode == 'SSD':
                scores['kmeans++'][mode] = evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode)
        timings['kmeans++'] = kmeans_time

        tmp = 0


    if 'tm' in methods:
        theta = 3.6

        tcentroids, tlabels, tm_time = theta_g2(points, theta=theta, num_shuffles=50)

        print('tm ', tm_time)
        print('Speedup', kmeans_time/tm_time)

        if verbose:
            plot_centroids_clusters(points, tcentroids, tlabels, 'tm', exp_type, initial_centroids=centroids)
        timings['tm'] = tm_time


    if 'birch' in methods:

        for bt in [2.0]:  # np.linspace(1.5, 2.5, 10):

            t1 = time()
            # birch = Birch(threshold=1.7, n_clusters=None).fit(points)
            # birch = Birch(threshold=1.7, n_clusters=int(np.prod(sz))).fit(points)
            # birch = Birch(threshold=3.0, n_clusters=int(np.prod(sz))).fit(points)
            birch = Birch(threshold=bt, n_clusters=int(np.prod(sz))).fit(points)

            t2 = time()

            birch_time = t2 - t1
            print('birch_time ', birch_time)

            birch_labels = birch.labels_
            birch_centroids = birch.subcluster_centers_
            print('# centroids ', len(birch_centroids))

            ext = str(np.round(bt, 2))
            if not show_thr:
                ext = ''

            if verbose:
                plot_centroids_clusters(points, birch_centroids, birch_labels, 'birch ' + ext, exp_type, initial_centroids=centroids)

            scores['birch ' + ext] = {}
            for mode in modes:
                if mode == 'SIL':
                    scores['birch ' + ext][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=birch_labels)
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['birch ' + ext][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=birch_labels, true_label=labels)
                elif mode == 'SSD':
                    scores['birch ' + ext][mode] = evaluate_centroids(birch_centroids, centroids, evaluation_thr, mode=mode)
            timings['birch ' + ext] = birch_time

    if 'mb_kmeans' in methods:

        t1 = time()
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=int(np.prod(sz)),
                            batch_size=int(np.prod(sz)),
                            n_init=50, max_no_improvement=10, verbose=0,
                            random_state=0).fit(points)
        t2 = time()
        mb_kmeans_time = t2 - t1
        print('mb_kmeans_time ', mb_kmeans_time)

        mbk_labels = mbk.labels_
        mbk_centroids = mbk.cluster_centers_
        print('# centroids ', len(mbk_centroids))
        if verbose:
            plot_centroids_clusters(points, mbk_centroids, mbk_labels, 'mini_batch', exp_type, initial_centroids=centroids)

        scores['mini_batch'] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['mini_batch'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=mbk_labels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['mini_batch'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=mbk_labels, true_label=labels)
            elif mode == 'SSD':
                scores['mini_batch'][mode] = evaluate_centroids(mbk_centroids, centroids, evaluation_thr, mode=mode)
        timings['mini_batch'] = mb_kmeans_time

    # pl.figure()
    # pl.plot(theta_range, elbow_check)
    # pl.grid()
    """
    pl.figure()
    pl.title('Score')
    x_tmp = np.arange(len(modes))
    width = 1/len(methods)

    for i, method in zip(range(len(scores.keys())), scores.keys()):
        pl.bar(x_tmp + width * i, scores[method].values(), width, label=method)
    pl.xticks(x_tmp + width / 2 + (len(methods)-2) * width/2 , modes)
    pl.legend()
    """

    fig = pl.figure()
    pl.title('Timings')
    x_tmp = np.arange(len(timings.values()))
    pl.bar(x_tmp, timings.values(), color=(1, 0, 0., 0.7))
    pl.xticks(x_tmp, timings.keys())
    pl.savefig('hard_Timing.png', dpi=300)
    pl.close(fig)

    print('hard timings', timings)