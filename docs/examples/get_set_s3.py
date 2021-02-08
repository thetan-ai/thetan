from time import time
import numpy as np
import string
import os
from numpy import testing as npt
from robotikos.theta import theta_g1, assign
from robotikos.fgm import fgn_shuffles, fgn
from robotikos.metric import IdentityFeature
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'S-sets')

def show_clusters(X, method, gt_centroids, labels, kcentroids, centroids):
    pl.figure()
    pl.title('Method is ' + method)
    unique_colors = np.random.rand(gt_centroids.shape[0], 3)

    colors = np.zeros((labels.shape[0], 3), dtype='f8')
    for i in range(X.shape[0]):
        colors[i] = unique_colors[labels[i]]
    pl.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'ko')
    pl.plot(kcentroids[:, 0], kcentroids[:, 1], 'r^')
    pl.plot(centroids[:, 0], centroids[:, 1], 'b^')
    pl.xlim([0 , 1000000])
    pl.savefig(os.path.join(dir_path,'s3_compare_GT_Tetha_KM++.png'), dpi=300)
    pl.show()



from pathlib import Path
points = np.loadtxt(os.path.join(dir_path ,'s3.txt'))
gt_centroids = np.loadtxt(os.path.join(dir_path ,'s3-cb.txt'))
labels_true =  np.loadtxt(os.path.join(dir_path ,'s3-label.txt')).astype('i8')
labels_true = labels_true - 1

points_orig = points.copy()

np.random.shuffle(points)

from theta_machines_tests import theta_g1, theta_g2, theta_g2x
n_clusters = 15

centroids, labels = theta_g1(points, theta=1000)
misses = {}
timings = {}
evaluation_thr = 0.5
np.random.seed(0)
t1 = time()
kmeans = KMeans(n_clusters=15, init='k-means++', n_init=10,
                max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(points)
t2 = time()
kmeans_time = t2 - t1
print('kmeans++ ', kmeans_time)
kcentroids = kmeans.cluster_centers_
klabels = kmeans.labels_
timings['kmeans++'] = kmeans_time
tmp = 0
print(kcentroids.shape)

theta_range_plot = False
metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
if theta_range_plot:
    range_theta = range(145000, 160000, 500)
    num_clusters = []
    for theta_t in range_theta:
        res = fgn_shuffles(points[:, None, :], thresholds=[(theta_t**2) / 2.0],
                           metric=metric,
                           num_shuffles=10, mode='fgn_ma', verbose=True)
        centroids = res[1]['centroids']
        num_clusters.append(len(centroids))
    pl.figure()
    pl.plot(range_theta, num_clusters)
    pl.plot(range_theta, num_clusters, 'o')
    pl.grid()
else:
    # centroids, labels = theta_g1(points, theta=145000)
    centroids, labels = theta_g2(points, theta=100000, num_shuffles=3000) ####s3
    unique_labels, sizes = np.unique(labels, return_counts=True)
    pl.figure()
    pl.plot(np.sort(sizes)[::-1], '-o')
    pl.show
    data = np.array(np.sort(sizes)[::-1])
    np.savetxt(os.path.join(dir_path,'s3_centroids_size.dat'), data)
    pl.savefig(os.path.join(dir_path,'s3_clusters_sizes_vs_numbers.png'), dpi=300)
    big_idx = np.argsort(sizes)[::-1][:15]
    big_centroids = centroids[big_idx]
    labels = assign(points, big_centroids)
    show_clusters(points_orig, 'S3: Compare gt labels with K-means++ and ThetA', gt_centroids, labels_true, kcentroids, big_centroids)
    labels.sort()
    klabels.sort()
    print('adjusted_rand_score for labels: ',adjusted_rand_score(labels_true, labels))
    print('normalized_mutual_info_score (labels_true, labels): ' , normalized_mutual_info_score (labels_true, labels))

    print('adjusted_rand_score for klabels: ',adjusted_rand_score(labels_true, klabels))
    print('normalized_mutual_info_score (labels_true, klabels): ' , normalized_mutual_info_score (labels_true, klabels))
