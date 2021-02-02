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

def show_clusters(X, method, centroids, gt_centroids, labels):
    pl.figure()
    pl.title('Method is ' + method)
    unique_colors = np.random.rand(centroids.shape[0], 3)
    colors = np.zeros((labels.shape[0], 3), dtype='f8')
    for i in range(X.shape[0]):
        colors[i] = unique_colors[labels[i]]
        # colors[:, 3] = 0.5
    pl.axis('equal')
    pl.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    # scatter(scentroids[:, 0], scentroids[:, 1], c=np.array([0.6, .6, .6]))
    pl.plot(centroids[:, 0], centroids[:, 1], 'bx')
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'm^')
    pl.show()


    # res = fgn_shuffles(points[:, None, :], thresholds=[(165000.0**2) / 2.0],
    #                    metric=metric,
    #                    num_shuffles=2000, mode='fgn_ma', verbose=True)
    # fcentroids = res[1]['centroids']
    # centroids = np.zeros((len(fcentroids), 2))
    # for i, c in enumerate(fcentroids.keys()):
    #     centroids[i] = np.squeeze(fcentroids[c])

from pathlib import Path

# dname = Path('c:/Users/elef/Devel/robotikos/docs/examples/S-Sets')
# dname = '/home/nasim/Devel/GRG-Lab/robotikos/docs/examples/S-sets/'
dname = 'C:/Users/nasim/Devel/GRG-Lab/robotikos/docs/examples/S-sets/'


########s1########
# points = np.loadtxt("S-sets\s1.txt")
# gt_centroids = np.loadtxt("S-sets\s1-cb.txt")
########s2########
# points = np.loadtxt("S-sets\s2.txt")
# gt_centroids = np.loadtxt("S-sets\s2-cb.txt")
########s2########

points = np.loadtxt(dname + "s4.txt")
gt_centroids = np.loadtxt(dname + "s4-cb.txt")
# points = np.loadtxt(dname / "s3.txt")
# gt_centroids = np.loadtxt(dname / "s3-cb.txt")
########s2########
# points = np.loadtxt("S-sets\s4.txt")
# gt_centroids = np.loadtxt("S-sets\s4-cb.txt")

np.random.shuffle(points)

from theta_machines_tests import theta_g1, theta_g2, theta_g2x
n_clusters = 15

centroids, labels = theta_g1(points, theta=1000)
methods = [] # ['kmeans++'] # 'kmeans', 'kmeans++', 'fgn_ma', 'birch', 'mb_kmeans', 'gmm']
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
    pl.scatter(points[:, 0], points[:, 1], alpha=0.5)
    pl.plot(kcentroids[:, 0], kcentroids[:, 1],'bx')
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1],'m^')
    pl.show()

theta_range_plot = False

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

if theta_range_plot:
    # range_theta = range(19835,970756,1000)
    # range_theta = range(0,6821,1000)
    # range_theta = range(100000, 200000, 500)
    range_theta = range(145000, 160000, 500)

    num_clusters = []
    for theta_t in range_theta:
        # centroids, labels = theta_g1(points, theta=theta_t)
        # centroids, labels = theta_g2(points, theta=theta_t, num_shuffles=50)

        res = fgn_shuffles(points[:, None, :], thresholds=[(theta_t**2) / 2.0],
                           metric=metric,
                           num_shuffles=10, mode='fgn_ma', verbose=True)
        centroids = res[1]['centroids']
        num_clusters.append(len(centroids))
    # pl.ylim([0 , 50])
    # pl.ylim([0 , 30])
    pl.figure()
    pl.plot(range_theta, num_clusters)
    pl.plot(range_theta, num_clusters, 'o')
    pl.grid()
    # indices = np.where(num_clusters == 15)[0]
    # print('1 p=', len(indices)/len(num_clusters))
    # pl.plot(range_theta[indices], num_clusters[indices], 'red')
    # y1 = 1
    # pl.fill_between(range_theta[indices], num_clusters[indices], facecolor='red', interpolate=True)
    pl.show()
else:
    # centroids, labels = theta_g2(points, theta=145000, num_shuffles=50) ####s1
    # centroids, labels = theta_g2(points, theta=148000, num_shuffles=50) ####s2
    # centroids, labels = theta_g2(points, theta=153000, num_shuffles=50) ####s3######
    # centroids, labels = theta_g1(points, theta=189162) ####s3_elef###### Not good
    # centroids, labels = theta_g2(points, theta=120000, num_shuffles=50) ####s4
    # centroids, labels = theta_g2(points, theta=680, num_shuffles=50) ####s1
    # centroids, labels = theta_g1(points, theta=120000) ####s3 - break problem into smaller problems

    # >>> Go to smaller clusters and find those that are dense
    # centroids, labels = theta_g2(points, theta=100000, num_shuffles=50) ####s3
    centroids, labels = theta_g2(points, theta=100000, num_shuffles=100) ####s3

    unique_labels, sizes = np.unique(labels, return_counts=True)
    # pl.plot(unique_labels, sizes, 'red')
    pl.plot(np.sort(sizes)[::-1], 'red')

    big_idx = np.argsort(sizes)[::-1][:15]

    big_centroids = centroids[big_idx]

    labels = assign(points, big_centroids)

    show_clusters(points, 'Big g2', big_centroids, gt_centroids, labels)



    # >>> Sub samples should be more sparse
    # sub_sample = 500
    # ss_index = np.random.randint(0, points.shape[0]-1, size = sub_sample)
    # points_sample = points[ss_index].copy()
    # points_sample = np.ascontiguousarray(points_sample)

    # final_centroids = []
    # for i in range(10):
    #     centroids, labels = theta_g2(points_sample, theta=100000, num_shuffles=50)
    #     final_centroids.append(centroids)

    # final_centroids = np.concatenate(final_centroids)
    # centroids, labels = theta_g2(final_centroids, theta=100000, num_shuffles=50)

    # labels = assign(points, centroids)

    # pl.figure()
    # pl.title('sample of ' + str(sub_sample))
    # pl.scatter(points[:sub_sample, 0], points[:sub_sample, 1], alpha=0.5)
    # pl.plot(gt_centroids[:, 0], gt_centroids[:, 1],'m^')
    # pl.show()



    # unique_labels, sizes = np.unique(labels, return_counts=True)
    print(centroids.shape)
    pl.figure()
    pl.title('All points')
    pl.scatter(points[:, 0], points[:, 1], alpha=0.5)
    #pl.scatter(kcentroids[:, 0], kcentroids[:, 1])
    pl.plot(centroids[:, 0], centroids[:, 1],'bx')
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1],'m^')
    pl.show()

    show_clusters(points, 'g1', centroids, gt_centroids, labels)

t =1
t =2