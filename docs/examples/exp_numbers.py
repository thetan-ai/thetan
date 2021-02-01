from time import time
import numpy as np
import string
from numpy import testing as npt
from robotikos.theta import theta_g1, assign
from robotikos.fgm import fgn_shuffles, fgn
from robotikos.metric import IdentityFeature
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

########s1########
# points = np.loadtxt("S-sets\s1.txt")
# gt_centroids = np.loadtxt("S-sets\s1-cb.txt")
########s2########
# points = np.loadtxt("S-sets\s2.txt")
# gt_centroids = np.loadtxt("S-sets\s2-cb.txt")
########s2########
points = np.loadtxt("S-sets\s3.txt")
gt_centroids = np.loadtxt("S-sets\s3-cb.txt")
########s2########
# points = np.loadtxt("S-set\s4.txt")
# gt_centroids = np.loadtxt("S-sets\s4-cb.txt")



from theta_machines_tests import theta_g1, theta_g2, theta_g2x
n_clusters = 15

centroids, labels = theta_g1(points, theta=1000)
methods = ['kmeans', 'kmeans++', 'fgn_ma', 'birch', 'mb_kmeans', 'gmm']
misses = {}
timings = {}
evaluation_thr = 0.5
if 'kmeans++' in methods:
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
    pl.scatter(points[:, 0], points[:, 1])
    pl.scatter(kcentroids[:, 0], kcentroids[:, 1])
    # pl.show()
theta_range_plot = True
if theta_range_plot is False:
    range_theta = range(19835,970756,1000)
    num_clusters = []
    for theta_t in range_theta:
        centroids, labels = theta_g1(points, theta=theta_t)
        # centroids, labels = theta_g2(points, theta=theta_t, num_shuffles=10)
        num_clusters.append(len(centroids))
    # pl.ylim([0 , 50])
    pl.ylim([0 , 30])
    pl.plot(range_theta, num_clusters)
    # indices = np.where(num_clusters == 15)[0]
    # print('1 p=', len(indices)/len(num_clusters))
    # pl.plot(range_theta[indices], num_clusters[indices], 'red')
    # y1 = 1
    # pl.fill_between(range_theta[indices], num_clusters[indices], facecolor='red', interpolate=True)s
    pl.show()
else:
    # centroids, labels = theta_g2(points, theta=145000, num_shuffles=50) ####s1
    # centroids, labels = theta_g2(points, theta=148000, num_shuffles=50) ####s2
    centroids, labels = theta_g2(points, theta=166000, num_shuffles=200) ####s3

    # centroids, labels = theta_g2(points, theta=166000, num_shuffles=200) ####s4



# unique_labels, sizes = np.unique(labels, return_counts=True)
    print(centroids.shape)
    pl.scatter(points[:, 0], points[:, 1])
    # pl.scatter(kcentroids[:, 0], kcentroids[:, 1])
    pl.plot(centroids[:, 0], centroids[:, 1],'x')
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1],'g^')
    pl.show()
