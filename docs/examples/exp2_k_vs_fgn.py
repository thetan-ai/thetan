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
from robotikos.fgm import fgn_shuffles, fgn_ma
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, rand_score, adjusted_rand_score
from robotikos.metricspeed import distance_matrix


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

    elif mode == 'RAI':
        score = adjusted_rand_score(true_label, label)
        #score = rand_score(true_label, label)

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


exp_type = 'blobs'
methods = ['kmeans', 'kmeans++', 'fgn_ma', 'birch', 'mb_kmeans', 'gmm']
modes = ['NMI', 'SIL', 'FMS', 'SSD', 'RAI']
evaluation_thr = 0.1
verbose = True
show_thr = False
radius = 4
n_local_samples = 25
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

plot_centroids_clusters(points, centroids, labels,
                                'original',
                                initial_centroids=centroids)


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
                                'kmeans',
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
                                'kmeans++',
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

if 'fgn_ma' in methods:

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

    elbow_check = []
    theta_range = [5.365]  # np.linspace(3.8, 5.5, num=40) # np.linspace(1, 3.5, num=50)
    theta_range = [7.25]  # with blobs 5.8 (104), 6.0 (101), 6.2 (102), 6.4 (101), 6.6 (102), 7.0 (101), 7.2 (101), 7.3 (99), 7.5 (98), 8.0 (96)
    theta_range = [6.0]
    # theta_range = [20., 10, 5.5]
    theta_range = [9.0]


    for fgn_t in theta_range: #np.linspace(5., 5.5, num=5):
        t1 = time()
        res = fgn_shuffles(points[:, None, :], thresholds=[fgn_t],
                           metric=metric,
                           num_shuffles=1, mode='fgn_ma', verbose=True)

        # res = fgn_ma(points[:, None, :], thresholds=theta_range, metric=metric,
        #              select_randomly=None, rng=None, verbose=True)

        t2 = time()

        fgn_time = t2 - t1
        print('fgn ', fgn_time)
        print('Speedup', kmeans_time/fgn_time)

        level = sorted(res.keys())[-1]
        try:
            fcentroids = res[level]['centroids']
            flabels = res[level]['labels']
        except TypeError:
            fcentroids_ = res[level].centroids
            fcentroids = {}
            for c in range(len(fcentroids_)):
                fcentroids[c] = fcentroids_[c]

            flabels = np.zeros(len(points), dtype='i8')
            for c in range(len(fcentroids_)):
                indices = res[level][c].indices
                flabels[indices] = c

        print('# centroids', len(fcentroids.keys()))
        elbow_check.append(len(fcentroids.keys()))

        fgn_centroids = np.zeros((len(fcentroids), 2))
        for i, c in enumerate(fcentroids.keys()):
            fgn_centroids[i] = np.squeeze(fcentroids[c])

        ext = str(np.round(fgn_t, 2))
        if not show_thr:
            ext = ''
        if verbose:
            plot_centroids_clusters(points, fgn_centroids, flabels, 'fgn ' + ext, initial_centroids=centroids)
        scores['fgn ' + ext] = {}
        for mode in modes:
            if mode == 'SIL':
                scores['fgn ' + ext][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=flabels)
            elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
                scores['fgn ' + ext][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=flabels, true_label=labels)
            elif mode == 'SSD':
                scores['fgn ' + ext][mode] = evaluate_centroids(fgn_centroids, centroids, evaluation_thr, mode=mode)
        timings['fgn ' + ext] = fgn_time


if 'birch' in methods:

    for bt in [1.94]:  # np.linspace(1.5, 2.5, 10):

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
                          n_init=10, max_no_improvement=10, verbose=0,
                          random_state=0).fit(points)
    t2 = time()
    mb_kmeans_time = t2 - t1
    print('mb_kmeans_time ', mb_kmeans_time)

    mbk_labels = mbk.labels_
    mbk_centroids = mbk.cluster_centers_
    print('# centroids ', len(mbk_centroids))
    if verbose:
        plot_centroids_clusters(points, mbk_centroids, mbk_labels, 'mini_batch', initial_centroids=centroids)

    scores['mini_batch'] = {}
    for mode in modes:
        if mode == 'SIL':
            scores['mini_batch'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=mbk_labels)
        elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
            scores['mini_batch'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=mbk_labels, true_label=labels)
        elif mode == 'SSD':
            scores['mini_batch'][mode] = evaluate_centroids(mbk_centroids, centroids, evaluation_thr, mode=mode)
    timings['mini_batch'] = mb_kmeans_time

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

    scores['gmm'] = {}
    for mode in modes:
        if mode == 'SIL':
            scores['gmm'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, X=points, label=gmm_labels)
        elif mode == 'NMI' or mode == 'FMS' or mode == 'RAI':
            scores['gmm'][mode] = evaluate_centroids(thr=evaluation_thr, mode=mode, label=gmm_labels, true_label=labels)
        elif mode == 'SSD':
            scores['gmm'][mode] = evaluate_centroids(gmm_centroids, centroids, evaluation_thr, mode=mode)
    timings['gmm'] = gmm_time

# pl.figure()
# pl.plot(theta_range, elbow_check)
# pl.grid()

pl.figure()
pl.title('Score')
x_tmp = np.arange(len(modes))
width = 1/len(methods)

for i, method in zip(range(len(scores.keys())), scores.keys()):
    pl.bar(x_tmp + width * i, scores[method].values(), width, label=method)
pl.xticks(x_tmp + width / 2 + (len(methods)-2) * width/2 , modes)
pl.legend()

pl.figure()
pl.title('Timings')
x_tmp = np.arange(len(timings.values()))
pl.bar(x_tmp, timings.values(), color=(1, 0, 0., 0.7))
pl.xticks(x_tmp, timings.keys())


pl.show()


t = 1

