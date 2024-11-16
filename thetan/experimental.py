from time import time
import tracemalloc
from threadpoolctl import threadpool_limits
import numpy as np
import pylab as pl
import pandas as pd
from thetan.berserker import berserker2, tsg_randomized, berserker2_clean
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, affinity_propagation
from sklearn.cluster import BisectingKMeans, MeanShift
from sklearn.cluster import Birch
from thetan.theta import distance_matrix as tdm
from thetan.theta import theta_g1 as tgs
from pyclustering.cluster.birch import birch
from pyclustering.cluster.bsas import bsas
from pyclustering.cluster.mbsas import mbsas
from pyclustering.cluster.ttsas import ttsas
from pyclustering.cluster.cure import cure
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.rock import rock
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.utils.metric import distance_metric, type_metric;
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
import skfuzzy as fuzz
from sklearn_extra.cluster import KMedoids

import matplotlib.pyplot as plt
from thetan.berserker import prepare_data_3d, X_to_many_3D, print_centroids, centroid_images


def pyclustering_utils(bsas_instance, repor=False, X=None):
    clusters = bsas_instance.get_clusters()
    type_repr = bsas_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)

    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()

    if repor is True:
        representors = bsas_instance.get_representors()
        centroids = bsas_instance.get_means()

    else:
        centroids = bsas_instance.get_representatives()
        # centroids = bsas_instance.get_means()

    labels = np.ascontiguousarray(clusters)
    centroids = np.ascontiguousarray(centroids)

    return labels, centroids


def clusters_to_3D_labels_custom(X, method, theta_value, data_dims, n_clusters=None, eps=None, min_samples=None, verbose=False, mask=None):
    """
    Deploying berserker v2 in imaging data and
    returning labels as 3D volumes.
    A mask can be provided in 1D form (ravelled)
    """

    if mask is not None:
        Xp = X[mask, :].copy()
        imask = 1 - mask
    else:
        Xp = X

    t_0 = time()
    if method.lower() == 'berserker2':
        centroids, labels = berserker2_clean(Xp, theta_value)

    if method.lower() == 'kmeans':

        with threadpool_limits(limits=1):
            kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1,
                            max_iter=300, tol=0.0001).fit(Xp)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

    if method.lower() == 'bsas':

        metric = distance_metric(type_metric.EUCLIDEAN)
        amount = n_clusters
        threshold = theta_value
        ccore = True

        bsas_instance = bsas(Xp, amount, threshold) # , ccore=ccore, metric=metric)
        bsas_instance.process()

        labels, centroids = pyclustering_utils(bsas_instance, X=Xp)


    if method.lower() == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X)
        unq = np.unique(model.labels_)
        db_nclusters = np.sum(unq>=0)
        centroids = np.array([np.mean(Xp[model.labels_==l], axis=0) for l in range(db_nclusters)])
        labels = model.labels_

    if method.lower() == 'ts':

        centroids, labels = tgs(Xp, theta_value, all_out=False, verbose=verbose)

    t_1 = time()

    if mask is not None:
        # We can calculate the centroids for label -1 on our own
        labelsp = np.zeros(np.prod(data_dims))
        labelsp[imask == 1] = -1
        labelsp[mask == 1] = labels.copy()
        centroid_neg1 = np.mean(X[imask], axis=0)
        centroids = np.vstack((centroid_neg1, centroids))
    else:
        labelsp = labels

    data_labels = labelsp.copy()
    data_labels = data_labels.reshape(data_dims)
    data_labels = data_labels.astype('f8')

    centroids_im = centroid_images(data_labels, centroids)
    return data_labels, centroids, centroids_im, t_1 - t_0
