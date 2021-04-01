from time import time
import numpy as np
import string
import os
from numpy import testing as npt
from thetan.theta import theta_g1, assign
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score


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
    pl.savefig(os.path.join(dir_path,'s1_compare_GT_TethA_KM++.png'), dpi=300)
    pl.show()

def show_points(X):
    pl.figure()
    pl.axis('equal')
    pl.title('Initial Data')
    pl.scatter(X[:, 0], X[:, 1], alpha=0.5)
    
def show_points_two(X, Y):
    pl.figure()
    pl.axis('equal')
    pl.title('Initial Data')
    pl.scatter(X[:, 0], X[:, 1], color='b', alpha=0.25)
    pl.scatter(Y[:, 0], Y[:, 1], color='g', alpha=0.25)

from pathlib import Path
points = np.loadtxt(os.path.join(dir_path,'s1.txt'))

show_points(points)


print(points.shape)

normal = np.random.normal(loc=0.0, scale=0.5, size=points.shape)

points2 = points * normal

points3 = points - 0.01 * points2


# show_points(points2)

show_points(points3)

# show_points(points4)

show_points_two(points, points3)

pl.show()