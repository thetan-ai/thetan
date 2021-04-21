from time import time
import numpy as np
import string
import os
from numpy import testing as npt
from thetan.theta import theta_g1, assign, gravity, calibrate
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'S-sets')


def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)

    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids


def grid_of_gaussians(n_local_samples, blob_dis=5):
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

    return X, gt_centroids


def show_clusters(X, method, gt_centroids, labels, kcentroids, centroids):
    pl.figure()
    pl.title('Method is ' + method)
    # unique_colors = np.random.rand(np.unique(labels).shape[0], 3)
    # Tricky point here if there if some labels are missing
    unique_colors = np.random.rand(labels.max() + 1, 3)

    colors = np.zeros((labels.shape[0], 3), dtype='f8')
    for i in range(X.shape[0]):
        colors[i] = unique_colors[labels[i]]
    pl.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'ko')
    # pl.plot(kcentroids[:, 0], kcentroids[:, 1], 'r^')
    pl.plot(centroids[:, 0], centroids[:, 1], 'b^')
    # pl.xlim([0 , 1000000])
    # pl.savefig(os.path.join(dir_path,'s1_compare_GT_TethA_KM++.png'), dpi=300)
    # pl.show()


def show_points(X, title='initial'):
    pl.figure()
    pl.axis('equal')
    pl.title(title)
    pl.scatter(X[:, 0], X[:, 1], alpha=0.1)


def show_points_two(X, Y, title='overlap'):
    pl.figure()
    pl.axis('equal')
    pl.title(title)
    pl.scatter(X[:, 0], X[:, 1], color='b', alpha=0.1)
    pl.scatter(Y[:, 0], Y[:, 1], color='g', alpha=0.1)


# X = np.loadtxt(os.path.join(dir_path, 's3.txt'))

X, gt_c = grid_of_gaussians(250, blob_dis=5)

np.random.shuffle(X)

X2 = X.copy()

# X = X[:int(0.1 * len(X))]

# X = np.ascontiguousarray(X)

# show_points(X)

theta = 4

centroids, labels, summs, clusters_sizes = theta_g1(X, theta=theta,
                                                    metric='euclidean',
                                                    all_out=True)

show_clusters(X, 'TSG', gt_c, labels, None, centroids)

# final_labels = gravity(X, centroids, clusters_sizes, theta/2., metric='euclidean')

def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)

    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids

# final_centroids = labels_to_centroids(X, final_labels)
# show_clusters(X, 'TSG-G', gt_c, final_labels, None, final_centroids)


def cluster_distribution(X, centroids, labels, clusters_sizes):

    distr = {}
    sumd = []
    sigd = []

    for c in range(centroids.shape[0]):
        distr[c] = []

    for i, l in enumerate(labels):
        distr[l].append(np.linalg.norm(X[i] - centroids[labels[i]]))

    pl.figure()
    pl.title('distr')
    max_length = 0
    for c in range(centroids.shape[0]):
        distr[c] = np.sort(distr[c])[::-1]
        current_length = len(distr[c])
        if current_length > max_length:
            max_length = current_length

    fdistr = np.zeros((centroids.shape[0], max_length), 'f8')
    integrals = []

    for c in range(centroids.shape[0]):
        tmp = np.zeros(max_length, 'f8')
        current_length = len(distr[c])
        tmp[:current_length] = distr[c]
        sumd.append(np.mean(distr[c]))
        sigd.append(np.std(distr[c]))
        distr[c] = tmp.copy()
        fdistr[c] = tmp.copy()
        integrals.append(np.sum(tmp))
        pl.plot(distr[c], alpha=0.2)
        # print(c, distr)

    pl.figure()
    pl.title('integrals')
    pl.plot(integrals, '.')

    return distr, sumd, fdistr, integrals


def cluster_distribution_hist(X, centroids, labels, theta):

    distr = {}

    for c in range(centroids.shape[0]):
        distr[c] = []

    for i, l in enumerate(labels):
        distr[l].append(np.linalg.norm(X[i] - centroids[labels[i]]))

    pl.figure()
    pl.title('distr - corrected')
    for c in range(centroids.shape[0]):
        distr[c], b = np.histogram(distr[c], bins=50, range=(0, 2*theta))
        # pl.plot(b[:-1], distr[c], alpha=0.2,)

        # pl.hist(distr[c], bins=50, range=(0, theta))
        # pl.show()

    arr = np.zeros((len(distr), distr[0].shape[0]))
    for c in range(centroids.shape[0]):
        arr[c] = distr[c]

    pl.figure()
    pl.title('distr - image')
    pl.imshow(arr)
    return distr, arr


distr, arr = cluster_distribution_hist(X, centroids, labels, theta)


# distr, sumd, fdistr, integrals = cluster_distribution(X, centroids, labels, clusters_sizes)

# theta2 = theta

# centroids2, labels2, summs2, clusters_sizes2 = theta_g1(fdistr, theta=theta2,
#                                                         metric='euclidean',
#                                                         all_out=True)
# print('Centroids2',  centroids2.shape)
# print('Fdistr',  fdistr.shape)

# integrals = np.array(integrals)[:, None]

# theta3 = np.mean(integrals) / 4

# centroids3, labels3, summs3, clusters_sizes3 = theta_g1(integrals, theta=theta3,
#                                                         metric='euclidean',
#                                                         all_out=True)
# print('Centroids3',  centroids3.shape)
# print('Labels3',  labels3.shape)
# print('Clusters_sizes3',  clusters_sizes3.shape)

# small_clusters_indices = np.where(clusters_sizes3 < np.mean(clusters_sizes3) / 2)[0]

# flagged = np.zeros(labels.shape, 'i8')
# for si in small_clusters_indices:
#     flagged[labels == si] = 1

# final_labels = calibrate(X, centroids, labels,
#                          flagged, metric='euclidean')

# final_centroids = labels_to_centroids(X, final_labels)

# show_clusters(X, 'TSG', gt_c, final_labels, None, final_centroids)




pl.show()

bkp = 1
