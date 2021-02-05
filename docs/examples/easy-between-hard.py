from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, adjusted_rand_score
from robotikos.metricspeed import distance_matrix
from theta_machines_tests import theta_g1, theta_g2, theta_g2x, labels_to_centroids
import seaborn as sns


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
    pl.xlabel('x')
    pl.ylabel('y')
    pl.savefig("C:/Users/nasim/OneDrive/Desktop/MyResearch/Compare_KM_++_ThetA.png", dpi=300)
    pl.show()
    # pl.clf()


exp_type = 'hard'
methods = ['fgn_ma']
modes = ['NMI', 'SIL', 'FMS', 'SSD', 'RAI']
evaluation_thr = 0.1
verbose = True
show_thr = False
radius = 4
n_local_samples = 100
blob_dis = 33
cluster_per_dim = 10
scores = {}
timings = {}

if exp_type == 'easy':
    # Generate centers for the blobs so that it forms a 5 X 5 grid.
    xx = np.linspace(-22, 22, 5)
    yy = np.linspace(-22, 22, 5)

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


if exp_type == 'between':
    # Generate centers for the blobs so that it forms a 5 X 5 grid.
    xx = np.linspace(-15, 15, 5)
    yy = np.linspace(-15, 15, 5)

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

if exp_type == 'hard':
    # Generate centers for the blobs so that it forms a 5 X 5 grid.
    xx = np.linspace(-10, 10, 5)
    yy = np.linspace(-10, 10, 5)
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


elbow_check = []
theta_range = np.arange(0., 50, 0.1)

number_of_clusters = []

for theta_t in theta_range:
    t1 = time()
    centroids, labels = theta_g2(points, theta_t, num_shuffles=100)

    t2 = time()

    tg_time = t2 - t1
    print('fgn ', tg_time)
    # print('Speedup', kmeans_time/fgn_time)

    print('# centroids', len(centroids))
    elbow_check.append(len(centroids))

    number_of_clusters.append(len(centroids))

    ext = str(np.round(theta_t, 2))
    if not show_thr:
        ext = ''
    if theta_t == 4:
        plot_centroids_clusters(points, centroids, labels, 'tg ' + ext, initial_centroids=gt_centroids)
    timings['tg ' + ext] = tg_time


pl.figure()
number_of_clusters = np.array(number_of_clusters)
pl.fill_between((theta_range), 0, number_of_clusters, facecolor='lightgrey', interpolate=True)
indices = np.where(number_of_clusters == 25)[0]
print('25 p=', len(indices)/len(number_of_clusters))
pl.plot((theta_range[indices]), number_of_clusters[indices], 'orange')
pl.fill_between((theta_range[indices]), 0, number_of_clusters[indices], facecolor='orange', interpolate=True)
pl.plot((theta_range), number_of_clusters, 'grey')
pl.xlabel(r'$\theta$')
pl.ylabel('Number of Clusters')
pl.ylim([0 , 50])
pl.xlim([0 , 50])
pl.show()
pl.savefig("C:/Users/nasim/OneDrive/Desktop/MyResearch/clusters_vs_ThetA_range.png", dpi=300)