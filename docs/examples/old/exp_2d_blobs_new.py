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
# from robotikos.metric import IdentityFeature
# from robotikos.metric import (NormalizedSumSquaredDiffMetric,
#                               NormalizedSumAbsDiffMetric)
from thetan.theta import theta_g1, assign
from thetan.theta import distance_matrix as tdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
# from robotikos.metricspeed import distance_matrix
from thetan.theta import thetan, thetan_density


def density_ratio(X, theta):

    centroid = np.mean(X, axis=0)

    d = np.linalg.norm(X - centroid, axis=1)

    return np.sum(d < theta/4), np.sum(d > theta/4)


def shake(summc, szc, theta, trials=2):

    or_meanc = summc / szc

    summc_tmp = summc.copy()

    for i in range(trials):
        summc_tmp += theta * np.random.randn(summc.shape[0])

    new_meanc = summc_tmp / (szc + trials)

    return or_meanc, new_meanc


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
        # metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
        # D = np.sqrt(distance_matrix(metric,
        #                             np.ascontiguousarray(ecentroids[:, None, :]),
        #                             np.ascontiguousarray(centroids[:, None, :])))

        D = np.sqrt(tdm(ecentroids, centroids))
        closest = np.min(D, axis=1)
        count = np.sum(closest < thr)
        score = count / len(ecentroids)

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


def plot_centroids_clusters(data, centroids, labels, title, exp_type, initial_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True):

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

    for c in range(centroids.shape[0]):
        pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

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
    # pl.savefig('2_'+exp_type+'_'+title+'.png', dpi=300)
    # pl.close(fig)




def plot_counts(counts, title):
    fig = pl.figure()
    pl.title(title)
    # pl.axis('equal')
    pl.plot(counts, '*')
    # pl.show()


def plot_nearc(nearc_value, nearc_dist, title):
    fig = pl.figure()
    pl.title(title)
    # pl.axis('equal')
    pl.plot(nearc_value, '.')
    pl.plot(nearc_dist, '.')
    # pl.show()


def plot_uncertainty_time(unc_time, title):
    fig = pl.figure()
    pl.title(title)
    pl.plot(unc_time)
    # pl.axis('equal')


exp_type = 'hard'
if exp_type == 'easy':
    methods = ['kmeans', 'kmeans++', 'tm', 'mb_kmeans', 'birch']
    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    evaluation_thr = 0.1
    verbose = False
    show_thr = False
    radius = 4
    n_local_samples = 30
    blob_dis = 33
    cluster_per_dim = 10
    scores = {}
    timings = {}

    if exp_type == 'blobs':

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
            local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)

            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)

        centroids = gt_centroids

    if exp_type == 'easy':

        blob_dis = 10
        cluster_x = 10
        cluster_y = 10
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

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)

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

        gt_centroids = []
        X = np.zeros((n_local_samples * n_centres.shape[0], 2))
        y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
        for i in range(n_centres.shape[0]):
            cpos = n_centres[i]

            local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)

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

                y1 = 5 * y + r * np.sin(theta)
                coords.append([x1, y1])

            centroids.append([5 * x, 5 * y])

        points = np.array(coords)
        centroids = np.array(centroids, dtype='f4')


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

        for bt in [2.7]:

            t1 = time()

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
    pl.title('2_'+'Timings')
    x_tmp = np.arange(len(timings.values()))
    pl.bar(x_tmp, timings.values(), color=(1, 0, 0., 0.7))
    pl.xticks(x_tmp, timings.keys())
    pl.savefig('2_'+'easy_Timing.png', dpi=300)
    pl.close(fig)
    print('easy timings', timings)

    t = 1
exp_type = 'hard'
if exp_type == 'hard':
    methods = ['kmeans', 'kmeans++', 'tm', 'mb_kmeans', 'birch']

    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    evaluation_thr = 0.1
    verbose = False
    show_thr = False
    radius = 4
    n_local_samples = 30
    blob_dis = 33
    cluster_per_dim = 10
    scores = {}
    timings = {}

    if exp_type == 'blobs':


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
            local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)

            X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
            y[i * n_local_samples: (i + 1) * n_local_samples] = i
            gt_centroids.append(local_X.mean(axis=0))

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)

        centroids = gt_centroids

    if exp_type == 'easy':

        blob_dis = 10 # It was 10
        cluster_x = 10
        cluster_y = 10
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

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)

        centroids = gt_centroids

    if exp_type == 'hard':
        blob_dis = 5 # It was 5
        cluster_x = 10
        cluster_y = 10
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

        gt_centroids = np.array(gt_centroids)

        points = X
        labels = y
        sz = len(n_centres)

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

                y1 = 5 * y + r * np.sin(theta)
                coords.append([x1, y1])

            centroids.append([5 * x, 5 * y])

        points = np.array(coords)
        centroids = np.array(centroids, dtype='f4')


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

        final_centroids, final_indices, final_summs, final_clusters_sizes, final_counts, final_nearc_value, final_nearc_dist = thetan(points, theta=theta, all_out=True)
        print('thetan')

        # find which clusters have most of the uncertainties
        plot_uncertainty_time((final_counts > 1).astype('f4'), 'uncertainty time')
        np.unique(final_indices)

        uncertain_points = points[final_counts > 1]
        plot_counts(final_counts, 'Percentage of greater than 1 neighbors ' + str(100 * np.sum(final_counts > 1) / final_counts.shape[0]))

        # plot_nearc(final_nearc_value, final_nearc_dist, 'Near centroind indices and distances')


        # if verbose:
        # plot_centroids_clusters(points, tcentroids, tlabels, 'tm', exp_type, initial_centroids=centroids)

        plot_centroids_clusters(points, final_centroids, final_indices, 'thetan', exp_type,
                                initial_centroids=centroids,
                                uncertain_data=uncertain_points, show=False)


        final_centroids2, final_indices2, final_summs2, final_clusters_sizes2, final_counts2, final_nearc_value2, final_nearc_dist2 = thetan(final_centroids, theta=theta, all_out=True)

        clusters = np.empty((final_centroids2.shape[0], 0)).tolist()
        for i in range(final_indices2.shape[0]):
            clusters[final_indices2[i]].append(i)

        candidates = []
        for c in clusters:
            if len(c) > 1:
                minc = np.min(c)
                candidates.append(minc)

        # We have to exclude the candidates and reassign their contents.
        candidatesX = []
        for i in range(final_indices.shape[0]):
            if final_indices[i] in candidates:
                candidatesX.append(i)
        X_candidates = points[candidatesX]

        final_centroids_reduced = final_centroids.copy()
        final_centroids_reduced = np.delete(final_centroids_reduced, candidates, axis=0)
        # Maybe We have to renumber the centroids.
        new_indices_candidates = assign(X_candidates, final_centroids_reduced)

        final_indices_updated = final_indices.copy()
        cnt = 0
        for i in range(final_indices.shape[0]):
            if final_indices[i] in candidates:
                final_indices_updated[i] = new_indices_candidates[cnt]
                cnt += 1

        # final_centroids_cp = final_centroids.copy()
        # np.random.shuffle(final_centroids_cp)
        # final_centroids2r, final_indices2r, final_summs2r, final_clusters_sizes2r, final_counts2r, final_nearc_value2r, final_nearc_dist2r = thetan(final_centroids_cp, theta=theta, all_out=True)

        # plot_centroids_clusters(points, final_centroids, final_indices_updated, 'thetan', exp_type,
        #                         initial_centroids=centroids,
        #                         uncertain_data=uncertain_points,
        #                         final_counts=final_counts,
        #                         final_nearc_value=final_nearc_value,
        #                         merged_centroids=final_centroids2)

        plot_centroids_clusters(points, final_centroids, final_indices, 'thetan1', exp_type,
                                initial_centroids=centroids,
                                uncertain_data=uncertain_points,
                                final_counts=final_counts,
                                final_nearc_value=final_nearc_value)


        # plot_centroids_clusters(points, final_centroids, final_indices_updated, 'thetan2', exp_type,
        #                         initial_centroids=centroids, show=False)

        # plot_centroids_clusters(points, final_centroids_reduced, final_indices_updated, 'thetan3', exp_type,
        #                         initial_centroids=centroids, show=True)


        # Apply thetan for each cluster and keep all centroids

        all_cl_centroids = []
        for i in range(centroids.shape[0]):
            X_tmp = points[final_indices == i].copy()
            cl_centroids, cl_indices, cl_summs, cl_clusters_sizes, cl_counts, cl_nearc_value, cl_nearc_dist = thetan(X_tmp, theta=theta, all_out=True)
            all_cl_centroids.append(cl_centroids)

        all_cl_centroids = np.concatenate(all_cl_centroids, axis= 0)

        np.random.shuffle(all_cl_centroids)

        plot_centroids_clusters(points, final_centroids, final_indices, 'thetan2', exp_type,
                                initial_centroids=centroids, uncertain_data=all_cl_centroids, show=True)

        f_centroids, f_indices, _, _ , _, _, _ = thetan(all_cl_centroids, theta=theta, all_out=True)

        plot_centroids_clusters(points, final_centroids, final_indices, 'thetan3', exp_type,
                                initial_centroids=centroids, uncertain_data=f_centroids, show=True)

        test_tmp = True
        timings['tm'] = tm_time



    if 'birch' in methods:

        for bt in [2.0]:

            t1 = time()

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



    fig = pl.figure()
    pl.title('Timings')
    x_tmp = np.arange(len(timings.values()))
    pl.bar(x_tmp, timings.values(), color=(1, 0, 0., 0.7))
    pl.xticks(x_tmp, timings.keys())
    pl.savefig('hard_Timing.png', dpi=300)
    pl.close(fig)

    print('hard timings', timings)
