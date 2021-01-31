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


#for exp_type, exp_param in zip(exps, exp_params):
#4.0, 12.0, 
#8, 24, 50, 76
for n_dim, theta, b_th, exp_type in zip([10, 100, 1000, 2000],
                                        [8, 24, 50, 76],
                                        [4.0, 12.0, 40.0, 75.0],
                                        ['high_dim', 'high_dim', 'high_dim', 'high_dim']):

    methods = ['kmeans', 'kmeans++', 'tm', 'birch', 'mb_kmeans']
    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    scores = {'Timing':[], 'method':[]}
    for mode in modes:
        scores[mode] = []

    for round_idx in range(10):
        #exp_type = 'hard'
        print('step', round_idx)
        evaluation_thr = 0.0005*n_dim
        verbose = False
        show_thr = False
        radius = 4
        n_local_samples = 20
        blob_dis = 33
        cluster_n = 100
        #scores = {}
        #timings = {}

        if exp_type == 'high_dim':
            # Generate centers for the blobs so that it forms a 10 X 10 grid.
            # xx = np.linspace(-22, 22, 10)
            # yy = np.linspace(-22, 22, 10)
            blob_dis = 10
            xx = np.arange(-1*blob_dis*cluster_n//2, blob_dis*cluster_n*2, blob_dis)
            xx = xx[:cluster_n]

            n_centres = np.pad(np.expand_dims(xx, -1), ((0, 0), (0, n_dim-1)), mode='edge')

            print(n_centres.shape)
            #X, y = make_blobs(n_samples=100000, centers=n_centres, random_state=0)
            # X, y = make_blobs(n_samples=100000, centers=n_centres, cluster_std = 1.0, random_state=0)

            gt_centroids = []
            X = np.zeros((n_local_samples * n_centres.shape[0], n_dim))
            y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
            for i in range(n_centres.shape[0]):
                cpos = n_centres[i]
                #local_X = cpos + radius*(np.random.rand(n_local_samples, 2) - 0.5)
                local_X = cpos + np.random.normal(0, 1, (n_local_samples, n_dim))
                X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
                y[i * n_local_samples: (i + 1) * n_local_samples] = i
                gt_centroids.append(local_X.mean(axis=0))

            gt_centroids = np.array(gt_centroids)

            points = X
            labels = y
            sz = len(n_centres)
            #centroids = np.array(n_centres, dtype='f4')
            centroids = gt_centroids

        points_idx = np.arange(np.shape(points)[0])
        np.random.shuffle(points_idx)
        new_points = np.zeros(np.shape(points))
        new_labels = np.zeros(np.shape(labels))
        for i, idx in zip(range(np.shape(points)[0]), points_idx):
            new_points[i] = points[idx]
            new_labels[i] = labels[idx]

        points = new_points
        labels = new_labels

        #print('After shuffling')
        #print(points[:10])

        print('Number of points ', points.shape)

        print('Number of gold centroids ', np.prod(sz))

        if 'kmeans' in methods:

            t1 = time()
            kmeans = KMeans(n_clusters=np.prod(sz), init='random', n_init=1,
                            max_iter=1000, tol=0.0001, random_state=0, n_jobs=1, precompute_distances=False, algorithm="auto").fit(points)
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
                elif mode == 'NMI' or mode == 'FMS':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(kmeans_time)

            tmp = 0


        if 'kmeans++' in methods:

            t1 = time()
            kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=1,
                            max_iter=1000, tol=0.0001, random_state=0, n_jobs=1, precompute_distances=False).fit(points)
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
                elif mode == 'NMI' or mode == 'FMS':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(kmeans_time)

            tmp = 0

        if 'tm' in methods:
            t1 = time()
            tcentroids, tlabels = theta_g1(points, theta=theta)
            tm_time = time() - t1
            print('tm ', tm_time)
            #print('Speedup', kmeans_time/tm_time)

            if verbose:
                plot_centroids_clusters(points, tcentroids, tlabels, 'tm', initial_centroids=centroids)

                #ext = str(np.round(fgn_t, 2))
            print(tcentroids.shape[0])
            scores['method'].append('tm')
            for mode in modes:
                if mode == 'SIL':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=tlabels))
                elif mode == 'NMI' or mode == 'FMS':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=tlabels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(tcentroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(tm_time)


        if 'birch' in methods:

            for bt in [b_th]:  # np.linspace(1.5, 2.5, 10):

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
                    plot_centroids_clusters(points, birch_centroids, birch_labels, 'birch ' + ext, initial_centroids=centroids)

                scores['method'].append('birch')
                for mode in modes:
                    if mode == 'SIL':
                        scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=birch_labels))
                    elif mode == 'NMI' or mode == 'FMS':
                        scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=birch_labels, true_label=labels))
                    elif mode == 'SSD':
                        scores[mode].append(evaluate_centroids(birch_centroids, centroids, evaluation_thr, mode=mode))
                scores['Timing'].append(birch_time)

        if 'mb_kmeans' in methods:

            t1 = time()
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=int(np.prod(sz)),
                                    batch_size=cluster_n,
                                    n_init=1, max_no_improvement=10, verbose=0,
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
                elif mode == 'NMI' or mode == 'FMS':
                    scores[mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=mbk_labels, true_label=labels))
                elif mode == 'SSD':
                    scores[mode].append(evaluate_centroids(mbk_centroids, centroids, evaluation_thr, mode=mode))
            scores['Timing'].append(mb_kmeans_time)

    """
    fig = pl.figure()
    pl.title('Hard_Timings_' + str(n_dim))
    pl.boxplot([timings[key] for key in timings.keys()], showfliers=False)
    pl.xticks([i for i in range(1, len(timings.keys())+1)], timings.keys())
    pl.savefig('hard_Timings_'+str(n_dim)+'_' + 'high_dim' + '.pdf')
    pl.close(fig)
    """
    data=pd.DataFrame(scores)
    boxplot = sns.boxplot(x="method", y="Timing", data=data, fliersize=0)
    fig = boxplot.get_figure()
    fig.savefig('easy_Timings_'+str(n_dim)+'_'+'high_dim.png', dpi=300)
    pl.close(fig)

    for mode in modes:
        boxplot = sns.boxplot(x="method", y=mode, data=data, fliersize=0)
        fig = boxplot.get_figure()
        fig.savefig('easy_'+mode + '_'+str(n_dim)+'_' + 'high_dim.png', dpi=300)
        pl.close(fig)

    """
    for mode in modes:
        fig = pl.figure()
        pl.title('Hard_' + mode + '_' + str(n_dim))
        pl.boxplot([scores[key][mode] for key in scores.keys()], showfliers=False)
        pl.xticks([i for i in range(1, len(scores.keys())+1)], scores.keys())
        pl.savefig(mode + '_'+str(n_dim)+'_' + 'high_dim' + '.pdf')
        pl.close(fig)
    """

    """
    for mode in modes:
        fig, (ax, ax2) = pl.subplots(2, 1, sharex=True)
        ax.boxplot([scores[key][mode] for key in scores.keys()])
        ax2.boxplot([scores[key][mode] for key in scores.keys()])
        ax.set_ylim(0.9, 1.0)
        ax2.set_ylim(0.0, 0.9)
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=False)
        ax2.xaxis.tick_bottom()

        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        pl.xticks([i for i in range(1, len(scores.keys())+1)], scores.keys())
        pl.savefig(mode + '_'+str(cluster_n)+'_' + 'cluster_n' + '.png')
        pl.close(fig)
    """