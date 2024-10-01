"""pip install scikit-learn-extra pandas scipy numpy pyclustering """
import os
from os.path import join as pjoin
from time import time
import tracemalloc
import numpy as np
import pylab as pl
import pandas as pd
from thetan.berserker import berserker2, tsg_randomized
from scipy.io import arff
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score, adjusted_rand_score
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
# import MeanShiftPP
import matplotlib
import matplotlib.pyplot as plt


col = ["#000000", "#DF536B", "#61D04F", "#2297E6", "#28E2E5", "#CD0BBC", "#F5C710"] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab10"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20b"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20c"].colors]

mrk = ["o", "^", "+", "x", "D", "v", "s", "*", "<", ">", "2"]


def parallel_bar(species, penguin_means, width, savefn=None):

    x = np.arange(len(species))  # the label locations
    # width = 0.25  # the width of the bars
    multiplier = 0

    # fig, ax = plt.subplots(layout='constrained')
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    fig.set_figheight(14)
    fig.set_figwidth(14)

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        measurement = np.round(measurement, 3)
        rects = ax.bar(x + offset, measurement, width, label=attribute.format('.3f'), alpha=0.5)
        ax.bar_label(rects, padding=3)
        ax.set_yscale("log")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Top performing')
    ax.set_xticks(x + width, species)
    # ax.legend(loc='upper left', ncols=4)
    ax.legend(loc='lower right', ncols=4)
    # ax.set_ylim(0, 250)
    # ax.grid()
    if savefn is None:
        plt.show()
    else:
        plt.savefig(savefn, dpi=300)


def pyclustering_utils(bsas_instance, repor=False):
    clusters = bsas_instance.get_clusters()
    type_repr = bsas_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)

    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()

    if repor is True:
        # representors = bsas_instance.get_representors()
        centroids = bsas_instance.get_means()

    else:
        centroids = bsas_instance.get_representatives()
        # centroids = bsas_instance.get_means()

    labels = np.ascontiguousarray(clusters)
    centroids = np.ascontiguousarray(centroids)

    return labels, centroids


def pyclustering_utils_kmedians(bsas_instance):
    clusters = bsas_instance.get_clusters()
    type_repr = bsas_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)

    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    centroids = bsas_instance.get_medians()

    labels = np.ascontiguousarray(clusters)
    centroids = np.ascontiguousarray(centroids)

    return labels, centroids


def pyclustering_utils_kmedoids(bsas_instance):
    clusters = bsas_instance.get_clusters()
    type_repr = bsas_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)

    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    centroids = bsas_instance.get_medoids()

    labels = np.ascontiguousarray(clusters)
    centroids = np.ascontiguousarray(centroids)

    return labels, centroids


def pyclustering_utils_xmeans(bsas_instance):
    clusters = bsas_instance.get_clusters()

    centroids = bsas_instance.get_centers()
    type_repr = bsas_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)

    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()

    labels = np.ascontiguousarray(clusters)
    centroids = np.ascontiguousarray(centroids)

    return labels, centroids


def pyclustering_utils_clarans(bsas_instance):
    clusters = bsas_instance.get_clusters()
    type_repr = bsas_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, X)

    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    medoids_indices = bsas_instance.get_medoids()

    labels = np.ascontiguousarray(clusters)
    # centroids = np.ascontiguousarray(centroids)

    return labels, medoids_indices


def calculate_nca(labels_true, labels_pred):
    """
    Calculate Normalized Clustering Accuracy (NCA) between two segmented images
    using the provided NCA formula.

    Parameters:
    - labels_true: numpy array of shape (H, W), ground truth labeled image.
    - labels_pred: numpy array of shape (H, W), predicted segmented image.

    Returns:
    - NCA: Normalized Clustering Accuracy.

    References
    ----------
    Gagolewski, M. (2024). Normalised clustering accuracy: An asymmetric external cluster validity measure. Journal of Classification. in press. URL: https://link.springer.com/content/pdf/10.1007/s00357-024-09482-2.pdf, DOI: 10.1007/s00357-024-09482-2.
    """
    if labels_true.shape != labels_pred.shape:
        raise ValueError("Labels must have the same dimensions for comparison.")

    # Flatten the images (if 2D) to 1D arrays for easier processing
    labels_true = labels_true.flatten()
    labels_pred = labels_pred.flatten()

    # Get the unique labels from both ground truth and predicted
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)

    # Calculate the number of clusters in both label sets
    num_clusters_true = len(unique_true)
    num_clusters_pred = len(unique_pred)

    # If cluster numbers differ, pad the smaller one with empty clusters
    max_clusters = max(num_clusters_true, num_clusters_pred)

    # Create a contingency matrix (matching table between clusters)
    contingency_matrix = np.zeros((max_clusters, max_clusters), dtype=int)

    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            # Count pixels where true_label corresponds to pred_label
            contingency_matrix[i, j] = np.sum((labels_true == true_label) & (labels_pred == pred_label))

    # If necessary, pad the matrix to match the largest number of clusters
    if num_clusters_true < max_clusters:
        padding = np.zeros((max_clusters - num_clusters_true, max_clusters), dtype=int)
        contingency_matrix = np.vstack((contingency_matrix, padding))

    if num_clusters_pred < max_clusters:
        padding = np.zeros((max_clusters, max_clusters - num_clusters_pred), dtype=int)
        contingency_matrix = np.hstack((contingency_matrix, padding))

    # Compute the total number of elements (total number of pixels/labels)
    total_elements = labels_true.size

    # Solve the linear assignment problem to find the best matching of clusters
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Compute NCA as per the given formula
    nca_sum = 0
    for j in range(max_clusters):
        i = row_ind[j]  # Corresponding true label index
        matching_size = contingency_matrix[i, col_ind[j]]

        # Ensure we handle the case where cluster sizes might be zero
        true_cluster_size = np.sum(contingency_matrix[i, :])
        pred_cluster_size = np.sum(contingency_matrix[:, col_ind[j]])

        # Avoid division by zero
        if true_cluster_size == 0 or pred_cluster_size == 0:
            continue

        # Following the NCA formula
        term1 = matching_size / true_cluster_size
        term2 = pred_cluster_size / total_elements
        nca_sum += (term1 - term2) / (1 - term2)

    nca = (1 / max_clusters) * nca_sum

    return nca


def evaluate(X, label, true_label, *, duration=-1, memory=None):

    try:
        nca_score = calculate_nca(true_label, label)
    except:
        nca_score = np.nan

    try:
        nmi_score = normalized_mutual_info_score(true_label, label)
    except:
        nmi_score = np.nan

    try:
        sil_score = silhouette_score(X, label)
    except:
        sil_score = np.nan

    try:
        fms_score = fowlkes_mallows_score(true_label, label)
    except:
        fms_score = np.nan

    nb_labels = f"{len(np.unique(label))}/{len(np.unique(true_label))}"

    return (nca_score, nmi_score, sil_score, fms_score, nb_labels,
            duration, memory)


def plot_centroids_clusters(
    data, centroids, labels, title, exp_type, initial_centroids=None,
    uncertain_data=None, final_counts=None, final_nearc_value=None,
    merged_centroids=None, show=True, dpath=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(data.shape[0]):
        new_colors[ind] = cluster_colors[labels[ind]]

    fig = pl.figure(figsize=(30, 15))
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')

    # for c in range(centroids.shape[0]):
    #    pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')

    if uncertain_data is not None:
        pl.plot(uncertain_data[:, 0], uncertain_data[:, 1], 'g.')

    cnt = 0
    if final_counts is not None:
        for i in range(final_counts.shape[0]):
            prev = cnt
            cnt += final_counts[i]
            if final_counts[i] > 1:
                pl.text(data[i, 0], data[i, 1], str(final_nearc_value[prev: cnt]) + ' ' + str(labels[i]), fontsize=8)

    if merged_centroids is not None:
        pl.plot(merged_centroids[:, 0], merged_centroids[:, 1], 'mo')

    if show:
        pl.show()

    pl.savefig(dpath + '2_' + exp_type + '_' + title + '.png', dpi=300)
    pl.close(fig)


def plot_dataset(X, labels, axis=None, title=None, xlabel=None, ylabel=None,
                 show=False, xlim=None, ylim=None, fname=None, **kwargs):
    """Draws a scatter plot of the labels."""
    if not isinstance(labels, (list, tuple)):
        labels = [labels, ]
    if not isinstance(title, (list, tuple)):
        title = [f"labels{i}" for i in range(len(labels))]

    p_args = {'projection': '3d'} if X.shape[1] == 3 else {}
    fig = pl.figure(figsize=(30, 15))
    for i, label in enumerate(labels):
        plt.subplot(1, len(labels), i+1, **p_args)
        plot_scatter(X, labels=label, axis="equal", title=title[i])  # , alpha=0.05)

    if show:
        plt.show()

    if not os.path.isdir(dpath):
        os.makedirs(dpath)

    pl.savefig(pjoin(dpath, 'rw_' + fname + '.png'), dpi=300)
    pl.close(fig)


def plot_scatter(X, y=None, labels=None, axis=None, title=None, xlabel=None,
                 ylabel=None, xlim=None, ylim=None, **kwargs):
    """Draws a scatter plot

    NOTE
    ----
    base on https://github.com/gagolews/genieclust/blob/master/genieclust/plots.py
    """
    X = np.array(X)
    if X.ndim == 2:
        if X.shape[1] not in [2, 3]:
            raise ValueError("`X` must have 2 or 3 columns.")
        if y is not None:
            raise ValueError("If `X` is a matrix, `y` should not be given.")
    elif X.ndim == 1:
        if y is None:
            raise ValueError("If `X` is a vector, `y` should be provided.")

        y = np.array(y)
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("`X` and `y` must have the same shape.")
        X = np.column_stack((X, y))
    else:
        raise ValueError("Invalid `X`.")

    if labels is None:
        labels = np.repeat(0, X.shape[0])
    else:
        labels = np.array(labels)
    if labels.ndim != 1 or X.shape[0] != labels.shape[0]:
        raise ValueError("Incorrect shape of `labels`.")

    for i in np.unique(labels):  # 0 = black, 1 = red, etc.
        z = {'s': X[labels == i, 2]} if X.shape[1] == 3 else {}
        plt.scatter(
            X[labels == i, 0],
            X[labels == i, 1],
            c=col[(i) % len(col)],
            marker=mrk[(i) % len(mrk)],
            **z,
            **kwargs
        )

    if axis is not None:
        plt.axis(axis)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)


def convert_str_labels_to_int(labels):
    # Decode byte strings to normal strings
    labels_str = np.array([label.decode('utf-8') for label in labels])

    # Create a label encoder by mapping unique strings to integers
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labels_str))}

    # Convert the string labels to numeric labels
    labels = np.array([label_mapping[label] for label in labels_str])

    return np.ascontiguousarray(labels, dtype='i8'), label_mapping


def get_real_world_data_names():
    data_path = pjoin(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data',
                      'real-world')
    data_path = os.path.abspath(data_path)
    data_names = [f for f in os.listdir(data_path)
                  if os.path.isfile(pjoin(data_path, f))
                  if f.endswith(".arff")]
    return sorted(data_names)


def get_real_world_data(*, filename='haberman.arff'):
    data_path = pjoin(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data',
                      'real-world')
    data_path = os.path.abspath(data_path)
    data, meta = arff.loadarff(pjoin(data_path, filename))
    df = pd.DataFrame(data)
    X = df.values[:, :-1]
    L = df.values[:, -1]
    nb_clusters = len(np.unique(L))

    if filename == 'haberman.arff':
        X = np.ascontiguousarray(X, dtype='f8')
        L = np.ascontiguousarray(L, dtype='i8')
        theta_value = 22.
    elif filename == 'iris.arff':
        X = np.ascontiguousarray(X, dtype='f8')
        L, mapping = convert_str_labels_to_int(L)
        theta_value = 1.75
    elif filename == 'arrhythmia.arff':
        X = np.ascontiguousarray(X, dtype='f8')
        L, mapping = convert_str_labels_to_int(L)
        theta_value = 341
    elif filename == 'balance-scale.arff':
        X = np.ascontiguousarray(X, dtype='f8')
        L, mapping = convert_str_labels_to_int(L)
        theta_value = 3.
    elif filename == 'sonar.arff':
        X = np.ascontiguousarray(X, dtype='f8')
        L, mapping = convert_str_labels_to_int(L)
        theta_value = 1.7

    return X, L, nb_clusters, theta_value


rows = ['TB', 'kmeans++', 'EM_GMM', 'TS', 'bisecting', 'dbscan',  #
        'hdbscan', 'kmeans', 'FCM', ]
# rows += ['kmedians', 'xmeans']  # kmedoids and hierarchical clustering removed due to large memory needs
rows += ['OPTICS', 'CLARANS']  # , 'meanshift++', 'TSR', 'meanshift', 'BSAS', 'MBSAS', 'TTSAS', 'BIRCH', 'CURE', 'TBK']

cols = ['NCA', 'NMI', 'SIL', 'FMS', 'Clusters', 'Duration', 'Memory']

dpath = pjoin(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'output',
              'real_world')

verbose = True
# theta_value = 22.
run_experiments = True
save_results = run_experiments
summarize_results = True
generate_plots = False

selected_data = ['haberman.arff', 'iris.arff', 'sonar.arff',
                 'balance-scale.arff', 'arrhythmia.arff']


if run_experiments:
    t_start = time()

    for rw_data_name in get_real_world_data_names():

        if 'haberman' not in rw_data_name:
            continue
        # if rw_data_name not in selected_data:
        #     continue

        print('######################################### > Experiment ', rw_data_name, ' Go Go Team Thetan!')

        evaluations = {}
        evaluation_thr = 0.1

        X, L, n_clusters, theta_value = get_real_world_data(filename=rw_data_name)

        print('Number of features ', X.shape[1])
        print('Number of Clusters ', n_clusters)
        print('Size in Bytes', X.nbytes / 1024 ** 2)
        print('theta_value ', theta_value)

        if 'TB' in rows:

            print('\n >> Starting Berserker ...')

            t = time()
            tracemalloc.start()
            centroids, labels = berserker2(X, theta_value, all_out=False)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration is ', duration)

            # if n_features == 2 or 3, we can plot
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['TB', 'GT'], fname='TB_'+rw_data_name, show=True, dpath=dpath)

            print('Starting evaluation')
            scores = evaluate(
                X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['TB'] = scores
            import ipdb; ipdb.set_trace()

        if 'kmeans++' in rows:

            print('\n >> Starting K-means++ ...')
            t = time()
            tracemalloc.start()
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
                            max_iter=300, tol=0.0001).fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            kcentroids = kmeans.cluster_centers_
            klabels = kmeans.labels_

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=klabels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['kmeans++'] = scores
            # if n_features == 2 or 3, we can plot
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['kmeans++', 'GT'], fname='kmeanspp_'+rw_data_name, show=False, dpath=dpath)

        if 'kmeans' in rows:

            print('\n >> Starting K-means (random) ...')
            t = time()
            tracemalloc.start()
            kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1,
                            max_iter=300, tol=0.0001).fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            kcentroids = kmeans.cluster_centers_
            klabels = kmeans.labels_

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=klabels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['kmeans'] = scores
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['kmeans', 'GT'], fname='kmeans_'+rw_data_name, show=False, dpath=dpath)

        if 'FCM' in rows:
            print('\n >> Starting Fuzzy C-means ...')
            # for expon in [2., 2.2, 3.]:
            expon = 2.
            print('Expon ', expon)
            t = time()
            tracemalloc.start()
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, n_clusters, expon, error=0.005, maxiter=1000, init=None)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            centroids = cntr
            labels = np.argmax(u, axis=0)

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['FCM', 'GT'], fname='FCM_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['FCM'] = scores

        # agglo = False # Due to large memory requirements - it kills my computer

        # if agglo:

        #     memory_dir = r"C:\\Users\\Eleftherios\\Devel\\data\\my_cached_memory_folder"
        #     model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", memory="cachedir")
        #     ac = model.fit(X)
        #     agg_centroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(model.n_clusters)])
        #     agg_labels = model.labels_

                # plot_dataset(
                #     X, [labels, L], title=['agglo', 'GT'], fname='agglo_'+rw_data_name, show=False, dpath=dpath)

        if 'meanshift' in rows:

            # for bandwidth in [2.4, 2.6, 2.8, 3.0, 3.2, 3.4]:
            print('\n >> Starting with MeanShift ...')
            t = time()
            bandwidth = 2.6
            print('bandwidth ', bandwidth)
            tracemalloc.start()
            model = MeanShift(bandwidth=bandwidth, cluster_all=True)
            ms = model.fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            ms_centroids = model.cluster_centers_ # np.array([np.mean(X[model.labels_==l], axis=0) for l in range(model.n_clusters)])
            ms_labels = model.labels_


            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=ms_labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['meanshift'] = scores

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['Meanshift', 'GT'], fname='meanshift_'+rw_data_name, show=False, dpath=dpath)


        if 'meanshift++' in rows:

            # for bandwidth in [2.6, 3.0, 3.2, 3.4, 3.6, 3.8]:
            print('\n >> Starting with MeanShift++ ...')
            t = time()
            bandwidth = 3.4
            print('bandwidth ', bandwidth)
            tracemalloc.start()
            model = MeanShiftPP.MeanShiftPP(bandwidth=bandwidth, threshold=0.00001, iterations=1000)
            cluster_centers_, labels_ = model.fit_predict(X, True)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            ms_centroids = np.ascontiguousarray(cluster_centers_, dtype='f8')
            ms_labels = labels_


            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=ms_labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['meanshift++'] = scores

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['MeanShift++', 'GT'], fname='meanshiftpp_'+rw_data_name, show=False, dpath=dpath)


        if 'EM_GMM' in rows:

            print('\n >> Starting EM_GMM .....')
            t = time()
            cov_type = ['spherical', 'diag', 'tied', 'full']
            tracemalloc.start()
            model = GaussianMixture(n_components=n_clusters,
                                    covariance_type=cov_type[3],
                                    max_iter=200, n_init=5, random_state=0)
            gmm = model.fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration ', duration)

            gmm_centroids = gmm.means_
            gmm_labels = gmm.predict(X)

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=gmm_labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time() - t)
            evaluations['EM_GMM'] = scores

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['EM_GMM', 'GT'], fname='EM_GMM_'+rw_data_name, show=False, dpath=dpath)

        if 'TS' in rows:

            print('\n >> Starting TS ...')

            t = time()
            tracemalloc.start()
            tgs_centroids, tgs_labels = tgs(X, theta_value, all_out=False)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration is ', duration)

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=tgs_labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time() - t)
            evaluations['TS'] = scores

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['TS', 'GT'], fname='TS_'+rw_data_name, show=False, dpath=dpath)

        if 'TSR' in rows:

            print('\n >> Starting TSR ...')
            t = time()
            tracemalloc.start()
            tgs_centroids, tgs_labels, tsg_sot = tsg_randomized(X, theta_value, num_shuffles=10)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration is ', duration)

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=tgs_labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time() - t)
            evaluations['TSR'] = scores

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['TSR', 'GT'], fname='TSR_'+rw_data_name, show=False, dpath=dpath)

        if 'bisecting' in rows:

            print('\n >> Starting Bisection K-means++ ...')
            t = time()
            tracemalloc.start()
            bkmeans = BisectingKMeans(n_clusters=n_clusters, n_init=1).fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            bkcentroids = bkmeans.cluster_centers_
            bklabels = bkmeans.labels_

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=bklabels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['bisecting'] = scores
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['Bisecting', 'GT'], fname='Bisecting_'+rw_data_name, show=False, dpath=dpath)

        eps = 0.5
        min_samples = 1  #40

        if 'dbscan' in rows:

            print('\n >> Starting DBSCAN ...')

            t = time()
            tracemalloc.start()
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            unq = np.unique(model.labels_)
            db_nclusters = np.sum(unq>=0)

            dbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(db_nclusters)])
            dblabels = model.labels_

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=dblabels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['dbscan'] = scores
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['DBScan', 'GT'], fname='DBScan_'+rw_data_name, show=False, dpath=dpath)


        if 'hdbscan' in rows:

            print('\n >> Starting HDBSCAN ...')
            t = time()
            tracemalloc.start()
            model = HDBSCAN(min_samples=min_samples)
            model.fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            unq = np.unique(model.labels_)
            hdb_nclusters = np.sum(unq>=0)

            hdbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(hdb_nclusters)])
            hdblabels = model.labels_

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=hdblabels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['hdbscan'] = scores
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['HDBSAN', 'GT'], fname='HDBSCAN_'+rw_data_name, show=False, dpath=dpath)

        if 'OPTICS' in rows:

            print('\n >> Starting OPTICS ...')
            t = time()
            tracemalloc.start()
            model = OPTICS(min_samples=2)  # , xi=0.05, min_cluster_size=5)
            model.fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()
            duration = time() - t
            print('Duration', duration)

            unq = np.unique(model.labels_)
            optics_nclusters = np.sum(unq>=0)

            opticscentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(optics_nclusters)])
            opticslabels = model.labels_

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=opticslabels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['OPTICS'] = scores
            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['OPTICS', 'GT'], fname='OPTICS_'+rw_data_name, show=False, dpath=dpath)


        if 'BSAS' in rows:

            print('\n >> Starting BSAS ...')
            theta_value = 3.6
            t = time()
            tracemalloc.start()
            metric = distance_metric(type_metric.EUCLIDEAN)
            ccore = True

            # sample = read_sample(path);
            # print("Sample: ", path);

            amount = n_clusters
            threshold = theta_value

            bsas_instance = bsas(X, amount, threshold, ccore=ccore, metric=metric)
            bsas_instance.process()
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, centroids = pyclustering_utils(bsas_instance)

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['BSAS', 'GT'], fname='BSAS_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['BSAS'] = scores


        if 'MBSAS' in rows:

            print('\n >> Starting MBSAS ...')
            theta_value = 3.6
            t = time()
            tracemalloc.start()
            metric = distance_metric(type_metric.EUCLIDEAN)
            ccore = True

            amount = n_clusters
            threshold = theta_value

            bsas_instance = mbsas(X, amount, threshold, ccore=ccore, metric=metric)
            bsas_instance.process()
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, centroids = pyclustering_utils(bsas_instance)

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['MBSAS', 'GT'], fname='MBSAS_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['MBSAS'] = scores


        if 'TTSAS' in rows:

            print('\n >> Starting TTSAS ...')

            # for threshold1 in np.arange(3, 5, 0.25):
            #     for sp in np.arange(0.2, 0.6, 0.1):
                    # threshold2 = threshold1 + sp
            t = time()
            tracemalloc.start()
            metric = distance_metric(type_metric.EUCLIDEAN)
            # ccore = True there is an issue with this variable

            amount = n_clusters

            threshold1 = 3.6
            threshold2 = 4
            # threshold1 = 3.75
            # threshold2 = 4.15
            print(threshold1)
            print(threshold2)
            # ttsas_instance = ttsas(X.tolist(), amount, threshold1, threshold2, metric=metric);
            # ttsas_instance.process()
            ccore = True
            ttsas_instance = ttsas(X, threshold1, threshold2, ccore=ccore, metric=metric)
            ttsas_instance.process()

            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            clusters = ttsas_instance.get_clusters()
            representatives = ttsas_instance.get_representatives()
            centroids = np.ascontiguousarray(representatives)

            labels = np.zeros(X.shape[0])
            for i, c in enumerate(clusters):
                labels[np.asarray(c)] = i
            labels = labels.astype('i8')
            duration = time() - t
            print('Duration is ', duration)
            print('# centroids ', len(centroids))

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['TTSAS', 'GT'], fname='TTSAS_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['TTSAS'] = scores


        if 'BIRCH' in rows:

            # for i, theta_value in enumerate([2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]):
            print('\n >> Starting BIRCH ...')
            # theta_value = 2.5 # 2.0 creates too many clusters. 3.0 too little.
            # print(theta_value)
            t = time()
            tracemalloc.start()
            amount = n_clusters

            birch = Birch(threshold=theta_value, branching_factor=500, n_clusters=amount).fit(X)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels = birch.labels_
            centroids = birch.subcluster_centers_
            print('# centroids ', len(centroids))

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['Birch', 'GT'], fname='BIRCH_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['BIRCH'] = scores

            t = 1

        if 'CURE' in rows:

            print('\n >> Starting CURE ...')
            t = time()
            tracemalloc.start()
            metric = distance_metric(type_metric.EUCLIDEAN)
            ccore = True

            number_clusters = n_clusters
            number_represent_points = 1
            compression = 1.

            bsas_instance = cure(X, number_clusters, number_represent_points, compression, ccore)
            bsas_instance.process()
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, centroids = pyclustering_utils(bsas_instance, True)

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['CURE', 'GT'], fname='CURE'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['CURE'] = scores

        if 'kmedians' in rows:

            print('\n >> Starting kmedians ...')

            t = time()
            tracemalloc.start()
            metric = distance_metric(type_metric.EUCLIDEAN)
            ccore = True

            init_centers = random_center_initializer(X, n_clusters).initialize()
            init_centers = np.ascontiguousarray(init_centers)

            kmedians_instance = kmedians(X, init_centers, ccore=ccore, metric=metric)
            kmedians_instance.process()
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, centroids = pyclustering_utils_kmedians(kmedians_instance)

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['kmedians', 'GT'], fname='kmedians_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['kmedians'] = scores

        if 'kmedoids' in rows:

            print('\n >> Starting kmedoids ...')

            t = time()
            tracemalloc.start()

            kmedoids = KMedoids(n_clusters=n_clusters, random_state=0, init='k-medoids++').fit(X)


            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, centroids = kmedoids.labels_, kmedoids.cluster_centers_

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['kmedoids', 'GT'], fname='kmedoids_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['kmedoids'] = scores

        if 'xmeans' in rows:

            print('\n >> Starting xmeans ...')

            t = time()
            tracemalloc.start()
            metric = distance_metric(type_metric.EUCLIDEAN)
            ccore = True

            # init_centers = kmeans_plusplus_initializer(X, n_clusters).initialize()
            init_centers = random_center_initializer(X, n_clusters).initialize()

            init_centers = np.ascontiguousarray(init_centers)

            xmeans_instance = xmeans(X, init_centers, n_clusters, ccore=ccore, metric=metric)
            xmeans_instance.process()
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, centroids = pyclustering_utils_xmeans(xmeans_instance)

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['xmeans', 'GT'], fname='xmeans_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['xmeans'] = scores


        if 'CLARANS' in rows:

            print('\n >> Starting CLARANS ...')

            t = time()
            tracemalloc.start()
            # metric = distance_metric(type_metric.EUCLIDEAN)
            # ccore = True

            # # init_centers = kmeans_plusplus_initializer(X, n_clusters).initialize()
            # init_centers = random_center_initializer(X, n_clusters).initialize()

            # init_centers = np.ascontiguousarray(init_centers)

            # xmeans_instance = xmeans(X, init_centers, n_clusters, ccore=ccore, metric=metric)
            # xmeans_instance.process()

            clarans_instance = clarans(X.tolist(), n_clusters, 6, 4)
            clarans_instance.process()

            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            duration = time() - t
            print('Duration is ', duration)

            labels, medroid_indices = pyclustering_utils_clarans(clarans_instance)
            centroids = np.ascontiguousarray(X[medroid_indices])

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['CLARANS', 'GT'], fname='CLARANS_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(X=X, label=labels, true_label=L, duration=duration, memory=peakmem)
            print(scores)
            print('Duration is ', time()-t)
            evaluations['CLARANS'] = scores


        if 'TBK' in rows:

            print('\n >> Starting Berserker ...')
            # theta_value = 3.6
            t = time()
            tracemalloc.start()
            centroids, labels = berserker2(X, theta_value, all_out=False)
            duration1 = time() - t

            print('Duration is ', duration1)

            uniq, sizes = np.unique(labels, return_counts=True)

            biggest_k = np.argsort(sizes)[-n_clusters:]

            centroids_tmp = np.ascontiguousarray(centroids[biggest_k].copy())

            print('\n >> Starting K-means (TB) ...')
            t = time()
            kmeans = KMeans(n_clusters=n_clusters, init=centroids_tmp, n_init=1,
                            max_iter=300, tol=0.0001).fit(X)
            duration2 = time() - t
            print('Duration is', duration2)

            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem // (1024**2)
            tracemalloc.stop()

            kcentroids = kmeans.cluster_centers_
            klabels = kmeans.labels_

            if verbose and X.shape[1] in [2, 3]:
                plot_dataset(
                    X, [labels, L], title=['TBK', 'GT'], fname='TBK_'+rw_data_name, show=False, dpath=dpath)

            print('Starting evaluation')
            t = time()
            scores = evaluate(
                X=X, label=klabels, true_label=L, duration=duration1 + duration2,
                memory=peakmem)
            print(scores)
            print('Duration is ', time() - t)

            evaluations['TBK'] = scores

        print('ALL EVALUATIONS')
        print(evaluations)

        # evals = np.zeros((len(rows), len(cols)), dtype=object)

        # for i, row in enumerate(rows):
        #     for j, col in enumerate(cols):
        #         evals[i, j] = evaluations[row][j]

        # df = pd.DataFrame(evals, columns=cols)
        df = pd.DataFrame.from_dict(evaluations, orient='index', columns=cols)

        print('Printing dataframe of evaluations ... ' + rw_data_name)
        print(df)

        if save_results:
            df.to_excel(pjoin(dpath, "res_" + rw_data_name + ".xlsx"))
            print('Saving evaluations')

    t_end = time()

    minutes, seconds = divmod(t_end - t_start, 60)
    print(f'Total duration is {minutes} minutes {seconds} seconds')

t = 1


if summarize_results:
    files = []
    for rw_data_name in get_real_world_data_names():
        if rw_data_name not in selected_data:
            continue
        fname_ = pjoin(dpath, "res_" + rw_data_name + ".xlsx")
        files.append(fname_)
        print(pd.read_excel(fname_))
        print()

    df = pd.concat([pd.read_excel(f) for f in files])
    import ipdb; ipdb.set_trace()
    result = df.groupby('Unnamed: 0').agg(['mean', 'std'])
    print(result)

    result.to_excel(pjoin(dpath, "summary.xlsx"))


def gen_plot(summary, species, species_id, fname_save_ac, fname_save_time):

    ACs = []
    NMIs = []
    SILs = []
    FMSs = []
    CLUSTERs = []
    DURATIONs = []
    MEMORYs = []

    for sp in species_id:

        entire_row = summary.iloc[sp]

        method_mean = entire_row[1::2].to_numpy().astype('f8')  # [:4]
        method_std = entire_row[2::2].to_numpy().astype('f8')  # [:4]

        # method_mean[method_mean < 0 ] = 0

        ACs.append(method_mean[0])
        NMIs.append(method_mean[1])
        SILs.append(method_mean[2])
        FMSs.append(method_mean[3])
        CLUSTERs.append(method_mean[4])
        DURATIONs.append(method_mean[5])
        MEMORYs.append(method_mean[6])

    penguin_means = {'AC': ACs, 'NMI': NMIs, 'SIL': SILs, 'FMS': FMSs}

    penguin_means2 = {'Clusters': CLUSTERs, 'Duration': DURATIONs, 'Memory': MEMORYs}

    parallel_bar(species, penguin_means, 0.2, savefn=fname_save_ac)

    parallel_bar(species, penguin_means2, 0.2, savefn=fname_save_time)


if generate_plots:

    summary = pd.read_excel(pjoin(dpath, "summary.xlsx"))

    # Correction (berserker2_clean) should be used for memory studies
    summary.loc[10, 'Memory'] = 8
    print(summary)

    t = 2

    # parallel_bar(species, penguin_means)

    species = ['TB', 'TBK', 'CURE', 'kmeans++', 'meanshift', 'hdbscan']
    species_id = [10, 11, 5, 19, 21, 17]
    fname_save_ac = dpath + 'top_ac.png'
    fname_save_time = dpath + 'top_time.png'

    # fname_save_ac = None
    # fname_save_time = None

    gen_plot(summary, species, species_id, fname_save_ac, fname_save_time)

    species = ['TB', 'TS', 'BIRCH', 'BSAS', 'MBSAS', 'TTSAS']
    species_id = [10, 12, 2, 3, 8, 14]
    fname_save_ac = dpath + 'similar_ac.png'
    fname_save_time = dpath + 'similar_time.png'

    # fname_save_ac = None
    # fname_save_time = None

    gen_plot(summary, species, species_id, fname_save_ac, fname_save_time)

    print(summary.to_latex())


t = 3
