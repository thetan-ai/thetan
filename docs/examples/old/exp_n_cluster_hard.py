from time import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

import seaborn as sns

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




def plot_centroids_clusters(data, centroids, labels, title, initial_centroids=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(len(points)):
        new_colors[ind] = cluster_colors[labels[ind]]

    pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')
    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')


for cluster_n, theta, exp_type in zip([10, 100, 1000, 2000], [15.4, 15.4, 15.4, 15.4], ['high_dim', 'high_dim', 'high_dim', 'high_dim']):
    methods = ['kmeans', 'kmeans++', 'ThetA', 'birch', 'mb_kmeans']
    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    scores = {'Timing':[], 'method':[], 'NMI':[], 'SIL':[], 'FMS':[], 'SSD':[]}

    for round_idx in range(10):

        print('step', round_idx)
        evaluation_thr = 0.5
        verbose = False
        show_thr = False
        radius = 4
        n_local_samples = 20
        blob_dis = 33
        cluster_per_dim = 10
        n_dim = 100


        if exp_type == 'high_dim':

            blob_dis = 1.6
            xx = np.arange(-1*blob_dis*cluster_n//2, blob_dis*cluster_n//2, blob_dis)
            xx = xx[:cluster_n]

            n_centres = np.pad(np.expand_dims(xx, -1), ((0, 0), (0, n_dim-1)), mode='edge')

            print(n_centres.shape)


            gt_centroids = []
            X = np.zeros((n_local_samples * n_centres.shape[0], n_dim))
            y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
            for i in range(n_centres.shape[0]):
                cpos = n_centres[i]

                local_X = cpos + np.random.normal(0, 1, (n_local_samples, n_dim))
                X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
                y[i * n_local_samples: (i + 1) * n_local_samples] = i
                gt_centroids.append(local_X.mean(axis=0))

            gt_centroids = np.array(gt_centroids)

            points = X
            labels = y
            sz = len(n_centres)

            centroids = gt_centroids
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
            cluster_x = 100
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
            cluster_x = 100
            cluster_y = 100
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


        points_idx = np.arange(np.shape(points)[0])
        np.random.shuffle(points_idx)
        new_points = np.zeros(np.shape(points))
        new_labels = np.zeros(np.shape(labels))
        for i, idx in zip(range(np.shape(points)[0]), points_idx):
            new_points[i] = points[idx]
            new_labels[i] = labels[idx]

        points = new_points
        labels = new_labels


        print('Number of points ', points.shape)

        print('Number of gold centroids ', np.prod(sz))

        if 'kmeans' in methods:

            t1 = time()
            kmeans = KMeans(n_clusters=np.prod(sz), init='random', n_init=50,
                            max_iter=300, tol=0.0001, random_state=0, n_jobs=1, precompute_distances=False).fit(points)
            t2 = time()

            kmeans_time = t2 - t1
            print('kmeans ', kmeans_time)

            kcentroids = kmeans.cluster_centers_
            klabels = kmeans.labels_
            if verbose:
                plot_centroids_clusters(points, kcentroids, klabels,
                                        'kmeans',
                                        initial_centroids=centroids)

            scores['method'].append('kmeans')
            for mode in modes:
                if mode == 'SIL':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(kmeans_time)

            tmp = 0


        if 'kmeans++' in methods:

            t1 = time()
            kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=50,
                            max_iter=300, tol=0.0001, random_state=0, n_jobs=1, precompute_distances=False).fit(points)
            t2 = time()

            kmeans_time = t2 - t1
            print('kmeans++ ', kmeans_time)

            kcentroids = kmeans.cluster_centers_
            klabels = kmeans.labels_
            if verbose:
                plot_centroids_clusters(points, kcentroids, klabels,
                                        'kmeans++',
                                        initial_centroids=centroids)

            scores['method'].append('kmeans++')
            for mode in modes:
                if mode == 'SIL':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(kmeans_time)

            tmp = 0


        if 'ThetA' in methods:

            tcentroids, tlabels, tm_time = theta_g2(points, theta=theta, num_shuffles=50)

            print('tm ', tm_time)

            if verbose:
                plot_centroids_clusters(points, tcentroids, tlabels, 'ThetA', initial_centroids=centroids)

            print(tcentroids.shape[0])
            scores['method'].append('ThetA')
            for mode in modes:
                if mode == 'SIL':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=tlabels))
                elif mode == 'NMI' or mode == 'FMS':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=tlabels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(tcentroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(tm_time)


        if 'birch' in methods:

            for bt in [12.75]:

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
                    plot_centroids_clusters(points, birch_centroids, birch_labels, 'birch ' + ext, initial_centroids=centroids)

                scores['method'].append('birch')
                for mode in modes:
                    if mode == 'SIL':
                        scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=birch_labels))
                    elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                        scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=birch_labels, true_label=labels))
                    elif mode == 'SSD':
                        scores[mode].append(evaluate_centroids(birch_centroids, centroids, evaluation_thr, mode=mode))
                scores['Timing'].append(birch_time)

        if 'mb_kmeans' in methods:

            t1 = time()
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=int(np.prod(sz)),
                                    batch_size=cluster_n,
                                    n_init=50, max_no_improvement=10, verbose=0,
                                    random_state=0).fit(points)
            t2 = time()
            mb_kmeans_time = t2 - t1
            print('mb_kmeans_time ', mb_kmeans_time)

            mbk_labels = mbk.labels_
            mbk_centroids = mbk.cluster_centers_
            print('# centroids ', len(mbk_centroids))
            if verbose:
                plot_centroids_clusters(points, mbk_centroids, mbk_labels, 'mini_batch', initial_centroids=centroids)

            scores['method'].append('mb_kmeans')
            for mode in modes:
                if mode == 'SIL':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=mbk_labels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=mbk_labels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(mbk_centroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(mb_kmeans_time)

        if 'gmm' in methods:

            t1 = time()
            cov_type = ['spherical', 'diag', 'tied', 'full']
            gmm = GaussianMixture(n_components=int(np.prod(sz)),
                                    covariance_type=cov_type[3],
                                    max_iter=200, n_init=5, random_state=0).fit(points)
            t2 = time()
            gmm_time = t2 - t1
            print('# gmm_time ', gmm_time)

            gmm_centroids = gmm.means_
            gmm_labels = gmm.predict(points)

            if verbose:
                plot_centroids_clusters(points, gmm_centroids, gmm_labels, 'gmm', initial_centroids=centroids)

            for mode in modes:
                if mode == 'SIL':
                    scores['gmm'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=gmm_labels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['gmm'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=gmm_labels, true_label=labels))
                elif mode == 'SSD':
                    scores['gmm'][mode].append(evaluate_centroids(gmm_centroids, centroids, evaluation_thr, mode=mode))
            timings['gmm'].append(gmm_time)


    data = pd.DataFrame(scores)
    boxplot = sns.boxplot(x='method', y='Timing', data=data, fliersize=0)
    fig = boxplot.get_figure()
    fig.savefig('hard_Timings_'+str(cluster_n)+'_cluster_n.png', dpi=300)
    pl.close(fig)
    for mode in modes:
        boxplot = sns.boxplot(x='method', y=mode, data=data, fliersize=0)
        fig = boxplot.get_figure()
        fig.savefig('hard_'+mode+'_'+str(cluster_n)+'_cluster_n.png', dpi=300)
        pl.close(fig)

