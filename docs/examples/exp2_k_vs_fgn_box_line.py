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
for n_dim, exp_type in zip([10, 100, 1000, 2000], ['high_dim', 'high_dim', 'high_dim', 'high_dim']):
    methods = ['kmeans', 'kmeans++', 'tm', 'birch', 'mb_kmeans']
    modes = ['NMI', 'SIL', 'FMS', 'SSD']
    scores = {}
    timings = {}
    for method in methods:
        scores[method] = {}
        timings[method] = []
        for mode in modes:
            scores[method][mode] = []

    for round_idx in range(10):
        #exp_type = 'hard'
        print('step', round_idx)
        evaluation_thr = 0.1
        verbose = False
        show_thr = False
        radius = 4
        n_local_samples = 20
        blob_dis = 33
        cluster_n = 10
        #scores = {}
        #timings = {}

        if exp_type == 'high_dim':
            # Generate centers for the blobs so that it forms a 10 X 10 grid.
            # xx = np.linspace(-22, 22, 10)
            # yy = np.linspace(-22, 22, 10)
            blob_dis = np.sqrt(25/n_dim)
            xx = np.arange(-1*blob_dis*cluster_n//2, blob_dis*cluster_n//2, blob_dis)
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
            cluster_x = 100
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
            cluster_x = 100
            cluster_y = 100
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


        #print('Before shuffling')
        #print(points[:10])

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
            kmeans = KMeans(n_clusters=np.prod(sz), init='random', n_init=10,
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
            for mode in modes:
                if mode == 'SIL':
                    scores['kmeans'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['kmeans'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels))
                elif mode == 'SSD':
                    scores['kmeans'][mode].append(evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode))
            timings['kmeans'].append(kmeans_time)

            tmp = 0


        if 'kmeans++' in methods:

            t1 = time()
            kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=20,
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

            for mode in modes:
                if mode == 'SIL':
                    scores['kmeans++'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=klabels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['kmeans++'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=klabels, true_label=labels))
                elif mode == 'SSD':
                    scores['kmeans++'][mode].append(evaluate_centroids(kcentroids, centroids, evaluation_thr, mode=mode))
            timings['kmeans++'].append(kmeans_time)

            tmp = 0

        # Not usable yet
        if 'local_search++' in methods:

            t1 = time()
            kmeans = KMeans(n_clusters=np.prod(sz), init='k-means++', n_init=20,
                            max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)


            ls_centroids = kmeans.cluster_centers_

            n_iter = int(np.prod(sz) * np.log(np.log(np.prod(sz))))

            dis = np.zeros(len(points))
            dis[0] = np.sum(np.square(points[0] - kcentroids[klabels[0]]))
            for idx in range(1, len(points)):
                dis[idx] = np.sum(np.square(points[idx] - kcentroids[klabels[idx]])) + dis[idx-1]
            for _ in range(n_iter):
                rand_n = np.random.uniform()
                border = rand_n * dis[-1]
                sel_idx = np.argwhere(dis >= border)[0]
                min_idx = -1
                for c_idx in range(len(ls_centroids)):
                    temp_centroids = ls_centroids.copy()
                    temp_centroids[c_idx] = points[sel_idx]
                    temp_dis = np.zeros(len(points))
                    temp_dis[0] = np.min(np.sum(np.square([temp_centroids[i] - points[0] for i in range(len(temp_centroids))])))
                    for p_idx in range(1, len(points)):
                        temp_dis[p_idx] = np.min(np.sum(np.square([temp_centroids[i] - points[p_idx] for i in range(len(temp_centroids))]))) + temp_dis[p_idx-1]
                    if dis[-1] > temp_dis[-1]:
                        min_idx = c_idx
                        dis = temp_dis
                
                if min_idx != -1:
                    ls_centroids[min_idx] = points[sel_idx]

            ls_labels = np.zeros(len(points))
            for p_idx in range(len(points)):
                ls_labels[p_idx] = np.argmin(np.sum(np.square([ls_centroids[i] - points[p_idx] for i in range(len(ls_centroids))])))


            if verbose:
                plot_centroids_clusters(points, kcentroids, klabels,
                                        'local_search++',
                                        initial_centroids=centroids)

            t2 = time()
            local_search_time = t2 - t1
            print('local_search++ ', local_search_time)
            scores['local_search++'] = {}
            for mode in modes:
                if mode == 'SIL':
                    scores['local_search++'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=ls_labels)
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['local_search++'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=ls_labels, true_label=labels)
                elif mode == 'SSD':
                    scores['local_search++'][mode] = evaluate_centroids(ls_centroids, centroids, evaluation_thr, mode=mode)
            timings['local_search++'] = kmeans_time

            tmp = 0

        if 'tm' in methods:
            theta = 5.1

            tcentroids, tlabels, tm_time = theta_g2(points, theta=theta, num_shuffles=20)

            print('tm ', tm_time)
            print('Speedup', kmeans_time/tm_time)

            if verbose:
                plot_centroids_clusters(points, tcentroids, tlabels, 'tm', initial_centroids=centroids)

                #ext = str(np.round(fgn_t, 2))
            print(tcentroids.shape[0])
            for mode in modes:
                if mode == 'SIL':
                    scores['tm'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=tlabels))
                elif mode == 'NMI' or mode == 'FMS':
                    scores['tm'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=tlabels, true_label=labels))
                elif mode == 'SSD':
                    scores['tm'][mode].append(evaluate_centroids(tcentroids, centroids, evaluation_thr, mode=mode))
            timings['tm'].append(tm_time)


        if 'birch' in methods:

            for bt in [4.0]:  # np.linspace(1.5, 2.5, 10):

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

                for mode in modes:
                    if mode == 'SIL':
                        scores['birch'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=birch_labels))
                    elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                        scores['birch'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=birch_labels, true_label=labels))
                    elif mode == 'SSD':
                        scores['birch'][mode].append(evaluate_centroids(birch_centroids, centroids, evaluation_thr, mode=mode))
                timings['birch'].append(birch_time)

        if 'mb_kmeans' in methods:

            t1 = time()
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=int(np.prod(sz)),
                                    batch_size=100,
                                    n_init=20, max_no_improvement=10, verbose=0,
                                    random_state=0).fit(points)
            t2 = time()
            mb_kmeans_time = t2 - t1
            print('mb_kmeans_time ', mb_kmeans_time)

            mbk_labels = mbk.labels_
            mbk_centroids = mbk.cluster_centers_
            print('# centroids ', len(mbk_centroids))
            if verbose:
                plot_centroids_clusters(points, mbk_centroids, mbk_labels, 'mini_batch', initial_centroids=centroids)

            for mode in modes:
                if mode == 'SIL':
                    scores['mb_kmeans'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=mbk_labels))
                elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                    scores['mb_kmeans'][mode].append(evaluate_centroids(thr=evaluation_thr, mode=mode, label=mbk_labels, true_label=labels))
                elif mode == 'SSD':
                    scores['mb_kmeans'][mode].append(evaluate_centroids(mbk_centroids, centroids, evaluation_thr, mode=mode))
            timings['mb_kmeans'].append(mb_kmeans_time)

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

    fig, (ax, ax2) = pl.subplots(2, 1, sharex=True)
    ax.boxplot([timings[key] for key in timings.keys()], showfliers=False)
    ax2.boxplot([timings[key] for key in timings.keys()], showfliers=False)
    ax.set_ylim(np.min(timings['mb_kmeans'])/2, np.max(timings['mb_kmeans'])*2)
    ax2.set_ylim(0.0, np.max(timings['tm'])*2)
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

    pl.xticks([i for i in range(1, len(timings.keys())+1)], timings.keys())
    pl.savefig('Timings_'+str(cluster_n)+'_' + 'high_dim' + '.png')
    pl.close(fig)

    for mode in modes:
        pl.figure()
        pl.title(exp_type + ' ' + mode)
        pl.boxplot([scores[method][mode] for key in scores.keys()], showfliers=False)
        pl.xticks([i for i in range(1, len(scores.keys())+1)], scores.keys())
        pl.savefig(mode + '_'+str(cluster_n)+'_' + 'high_dim' + '.png')
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