"""Please, do:

pip install genieclust clustering-benchmarks

git submodule update --remote

dataset url: https://clustering-benchmarks.gagolewski.com/weave/data-v1.html#wut/circles

"""
from dataclasses import make_dataclass
import os
from os.path import join as pjoin
from pprint import pprint
import tracemalloc
import time
import warnings


import clustbench
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment
from sklearn import datasets, manifold
from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pair_confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from thetan.berserker import berserker2


DATA_PATH = pjoin(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data',
                  'clustering-data-v1')
DATA_PATH = os.path.abspath(DATA_PATH)
RESULTS_PATH = pjoin(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data',
                     'clustering-results-v1', 'original')
RESULTS_PATH = os.path.abspath(RESULTS_PATH)

col = ["#000000", "#DF536B", "#61D04F", "#2297E6", "#28E2E5", "#CD0BBC", "#F5C710"] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab10"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20b"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20c"].colors]

mrk = ["o", "^", "+", "x", "D", "v", "s", "*", "<", ">", "2"]

DataSet = make_dataclass(
    "ClusteringBenchmark",
    ["battery", "dataset", "description", "data", "labels", "n_clusters"]
)


def plot_dataset(dataset, axis=None, title=None, xlabel=None, ylabel=None,
                 start=0, xlim=None, ylim=None, **kwargs):
    """Draws a scatter plot of the labels."""
    if not isinstance(dataset, (list, tuple)):
        dataset = [dataset, ]

    nb_subplot = sum(len(data.labels) for data in dataset)
    if not isinstance(title, (list, tuple)):
        title = [f"labels{i}" for i in range(nb_subplot)]
    if len(title) != nb_subplot:
        title += [f"labels{i}" for i in range(nb_subplot-len(title))]
    for k, data in enumerate(dataset, 1):
        for i in range(len(data.labels)):
            plt.subplot(1, nb_subplot, k+i)
            plot_scatter(data.data, labels=data.labels[i]-1,
                         axis="equal", title=title[k+i-1])  # , alpha=0.05)
    plt.show()


def plot_scatter(X, y=None, labels=None, axis=None, title=None, xlabel=None,
                 ylabel=None, xlim=None, ylim=None, **kwargs):
    """Draws a scatter plot

    NOTE
    ----
    base on https://github.com/gagolews/genieclust/blob/master/genieclust/plots.py
    """
    X = np.array(X)
    if X.ndim == 2:
        if not X.shape[1] == 2:
            raise ValueError("`X` must have 2 columns.")
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
        plt.scatter(
            X[labels == i, 0],
            X[labels == i, 1],
            c=col[(i) % len(col)],
            marker=mrk[(i) % len(mrk)],
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


def get_battery_names():
    """Get the names of the batteries."""
    batteries = [f for f in os.listdir(DATA_PATH)
                 if os.path.isdir(pjoin(DATA_PATH, f))
                 if not f.startswith(".")]
    return sorted(batteries)


def get_dataset_names(*, battery_name='mnist'):
    """Get the names of the datasets."""
    dataset = [f.replace(".data.gz", "")
               for f in os.listdir(pjoin(DATA_PATH, battery_name))
               if os.path.isfile(pjoin(DATA_PATH, battery_name, f))
               if f.endswith(".data.gz")]
    return sorted(dataset)


def load_dataset(battery, dataset, *, preprocess=True, random_state=None):
    """Load a dataset from the clustering benchmark."""

    data_file = pjoin(DATA_PATH, battery, f"{dataset}.data.gz")
    data = np.loadtxt(data_file, ndmin=2)

    if data.ndim != 2:
        raise ValueError("Not a matrix.")

    if preprocess:
        # Normalise a data matrix

        # Removes all columns of zero variance (constant). Centres the data
        # around the centroid (so that each column mean is 0).
        # Scales all columns proportionally (so that the total variance is 1;
        # note that this is not the same as standardisation: standard
        # deviations in each column might still be different).
        # Adds a tiny amount of noise to minimise the risk of having
        # duplicate points.
        data = data[:, data.var(axis=0) > 0]

        # centre + scale all columns proportionally (not: standardise)
        data = (data-data.mean(axis=0))/data.std(axis=None, ddof=1)

        # add a tiny bit of white noise:
        noise_factor = 1e-6
        data += sp.stats.norm.rvs(
            loc=0.0, scale=data.std(ddof=1)*noise_factor, size=data.shape,
            random_state=random_state
        )

    labels, idx = [], 0
    while True:
        label_file = pjoin(DATA_PATH, battery, f"{dataset}.labels{idx}.gz")
        if not np.DataSource().exists(label_file):
            break  # stop trying to find more label vectors

        label = np.loadtxt(label_file, dtype="int")
        if label.ndim != 1 or label.shape[0] != data.shape[0]:
            raise ValueError(f"Incorrect number of labels in '{label_file}'.")

        labels.append(label)
        idx += 1

    n_clusters = np.array([np.max(ll) for ll in labels])

    desc_file = pjoin(DATA_PATH, battery, f"{dataset}.txt")
    with np.DataSource().open(desc_file, "r") as readme_file:
        description = readme_file.read()

    return DataSet(
        battery=battery,
        dataset=dataset,
        description=description,
        data=data,
        labels=labels,
        n_clusters=n_clusters,
    )


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


def rand_score(labels_true, labels_pred):
    """
    Compute the Rand Index (RI) between two clusterings.

    Parameters:
    - labels_true: Ground truth clustering (1D array).
    - labels_pred: Predicted clustering (1D array).

    Returns:
    - Rand Index (float).
    """
    # Compute the pair confusion matrix (YY, YN, NY, NN)
    YY, YN, NY, NN = pair_confusion_matrix(labels_true, labels_pred).ravel()
    n = YY + YN + NY + NN

    # Rand index formula
    rand_index = (YY + NN) / n
    return rand_index


def fowlkes_mallows_score(labels_true, labels_pred):
    """
    Compute the Fowlkes-Mallows Index (FM) between two clusterings.

    Parameters:
    - labels_true: Ground truth clustering (1D array).
    - labels_pred: Predicted clustering (1D array).

    Returns:
    - Fowlkes-Mallows Index (float).
    """
    # Compute the pair confusion matrix (YY, YN, NY, NN)
    YY, YN, NY, _ = pair_confusion_matrix(labels_true, labels_pred).ravel()

    # Fowlkes-Mallows formula
    fm_index = YY / np.sqrt((YY + YN) * (YY + NY))
    return fm_index


def adjusted_rand_score_custom(labels_true, labels_pred):
    """
    Compute the Adjusted Rand Index (ARI) between two clusterings.

    Parameters:
    - labels_true: Ground truth clustering (1D array).
    - labels_pred: Predicted clustering (1D array).

    Returns:
    - Adjusted Rand Index (float).
    """
    # Use scikit-learn's implementation of ARI
    return adjusted_rand_score(labels_true, labels_pred)


def normalized_mutual_information(labels_true, labels_pred):
    """
    Compute the Normalized Mutual Information (NMI) between two clusterings.

    Parameters:
    - labels_true: Ground truth clustering (1D array).
    - labels_pred: Predicted clustering (1D array).

    Returns:
    - Normalized Mutual Information (float).
    """
    # Use scikit-learn's implementation of NMI
    return normalized_mutual_info_score(labels_true, labels_pred)


def test_tb():
    csv_table = [('battery', 'dataset', 'theta', 'Original n_clusters', 'Final n_clusters', 'speed', 'nca', 'ri', 'fm', 'ari', 'nmi')]

    # test_data = [('wut', 'circles', 1.20), ('wut', 'z3', 1.1), ('sipu', 'birch1', 0.25),
    #              ('wut', 'mk1', 1.5), ('wut', 'trajectories', 1.3), ('wut', 'x1', 1.),
    #              ('sipu', 'a1', .3), ('sipu', 'r15', 1.), ('sipu', 'aggregation', .99),
    #             ('sipu', 'unbalance', .35), ('fcps', 'twodiamonds', 1.5)]
    # test_data = [('sipu', 'aggregation', .99), ('g2mg', 'g2mg_2_30', 1.4)]
    test_data = [('fcps', 'lsun', 1.65)]
    for battery, dataset, theta_value in test_data:
        csv_data = [battery, dataset, theta_value]
        dataset = load_dataset(battery=battery, dataset=dataset)
        # plot_dataset(dataset)
        print('Starting Berserker')
        t_start = time.perf_counter()
        centroids, labels = berserker2(np.ascontiguousarray(dataset.data),
                                       theta_value, all_out=False, verbose=False)
        speed = time.perf_counter() - t_start
        print('Duration is ', speed)
        print(f'cluster size: {len(centroids)} / {dataset.n_clusters} expected')
        results = DataSet(
            battery=dataset.battery,
            dataset=dataset.dataset,
            description=dataset.description,
            data=dataset.data,
            labels=[labels],
            n_clusters=[len(centroids)]
        )
        csv_data.extend([dataset.n_clusters[-1], len(centroids), speed])
        # res = clustbench.load_results('*', dataset.battery, dataset.dataset,
        #                               results.n_clusters, path=RESULTS_PATH
        #                               )
        # labels2 = labels + 1
        # all_scores = {method: clustbench.get_score(labels2, res[method]) for method in res.keys()}
        # pprint(all_scores)

        nca = calculate_nca(dataset.labels[-1], labels)
        ri = rand_score(dataset.labels[-1], labels)
        fm = fowlkes_mallows_score(dataset.labels[-1], labels)
        ari = adjusted_rand_score_custom(dataset.labels[-1], labels)
        nmi = normalized_mutual_information(dataset.labels[-1], labels)

        print(f"Normalized Clustering Accuracy: {nca}")
        print(f"Rand Index (RI): {ri}")
        print(f"Fowlkes-Mallows Index (FM): {fm}")
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Normalized Mutual Information (NMI): {nmi}")
        csv_data.extend([nca, ri, fm, ari, nmi])
        plot_dataset([results, dataset])
        csv_table.append(csv_data)
    np.savetxt('tb_results.csv', csv_table, delimiter=',', fmt='%s')


def test_tb_hdbscan():
    csv_table = [('Dataset', 'Final N_Clusters', 'Speed', 'NCA', 'RI', 'FM', 'ARI', 'NMI')]
    test_data = [('wut', 'trajectories', .1, 1., 0.3), ('wut', 'windows', .1, 1., 0.3),
                 ('other', 'square', .2, 1, 0.4), ('fcps', 'wingnut', .25, 1, 0.42),
                 ('graves', 'zigzag', .25, 1., 0.5), ('fcps', 'target', .1, 1., 0.5),
                 ('fcps', 'lsun', .1, 1., 0.35), ('wut', 'mk2', .05, 1, 0.35)
                 ]
    # test_data = [('graves', 'dense', .048)]
    # test_data = [('wut', 'x2', .05, 1, 0.35)]
    # test_data = [('graves', 'parabolic', .1, 1, 0.25)]
    # test_data = [('graves', 'dense', .6)]  # 2 / 0.6
    # test_data = [('graves', 'dense', .6)]  # 2 / 0.6
    # test_data = [('graves', 'fuzzyx', .25, 1, .36)]  # 2 / 0.6
    # test_data = [('wut', 'labirynth', .25, 1, .2)]  # 2 / 0.6
    test_data = [('fcps', 'wingnut', .25, 1, 0.42)]  # 2 / 0.6
    for battery, dataset, theta_value, min_samples, eps in test_data:
        dataset = load_dataset(battery=battery, dataset=dataset)
        # plot_dataset(dataset)
        print('Starting Berserker')
        t_start = time.perf_counter()
        centroids, labels = berserker2(np.ascontiguousarray(dataset.data),
                                       theta_value, all_out=False, verbose=False)
        print('Duration is ', time.perf_counter() - t_start)
        print(f'cluster size: {len(centroids)} / {dataset.n_clusters} expected')
        results = DataSet(
            battery=dataset.battery,
            dataset=dataset.dataset,
            description=dataset.description,
            data=dataset.data,
            labels=[labels],
            n_clusters=[len(centroids)]
        )
        # plot_dataset([results, dataset], title=['TB', 'Ground Truth'])
        print('Starting HDBSCAN')
        min_samples = int(min_samples)
        cluster_selection_epsilon = 1.
        # model = HDBSCAN(min_samples=min_samples,
        #                 cluster_selection_epsilon=cluster_selection_epsilon)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(centroids)
        speed = time.perf_counter() - t_start
        print('Duration is ', speed)
        unq = np.unique(model.labels_)
        hdb_nclusters = np.sum(unq >= 0)
        csv_data = [dataset.dataset, f"{hdb_nclusters}/{dataset.n_clusters[0]}", f"{speed:.8f}"]
        print(f'cluster size: {hdb_nclusters} / {dataset.n_clusters} expected')
        final_labels = labels.copy()
        for i, l in enumerate(labels):
            final_labels[i] = model.labels_[l]

        results = DataSet(
            battery=dataset.battery,
            dataset=dataset.dataset,
            description=dataset.description,
            data=dataset.data,
            labels=[final_labels],
            n_clusters=[hdb_nclusters]
        )
        # res = clustbench.load_results('*', dataset.battery, dataset.dataset,
        #                               results.n_clusters, path=RESULTS_PATH
        #                               )
        # labels2 = final_labels + 1
        # all_scores = {method: clustbench.get_score(labels2, res[method]) for method in res.keys()}
        # pprint(all_scores)
        nca = calculate_nca(dataset.labels[-1], results.labels[0])
        ri = rand_score(dataset.labels[-1], results.labels[0])
        fm = fowlkes_mallows_score(dataset.labels[-1], results.labels[0])
        ari = adjusted_rand_score_custom(dataset.labels[-1], results.labels[0])
        nmi = normalized_mutual_information(dataset.labels[-1], results.labels[0])

        print(f"Normalized Clustering Accuracy: {nca}")
        print(f"Rand Index (RI): {ri}")
        print(f"Fowlkes-Mallows Index (FM): {fm}")
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Normalized Mutual Information (NMI): {nmi}")

        plot_dataset([results, dataset], title=['TB-DBSCAN', 'Ground Truth'])
        csv_data.extend([f"{nca:.5f}", f"{ri:.5f}", f"{fm:.5f}", f"{ari:.5f}", f"{nmi:.5f}"])
        csv_table.append(csv_data)

    np.savetxt('tb_hdbscan_results.csv', csv_table, delimiter=',', fmt='%s')


def test_tb_tb():
    # test_data = [('wut', 'trajectories', .1), ('wut', 'windows', .1),
    #              ('other', 'square', .8), ('fcps', 'wingnut', .5),
                #    ('graves', 'zigzag', .25), ('fcps', 'target', .1),
                #    ('fcps', 'lsun', .1)
    #              ]
    # test_data = [('graves', 'dense', .048)]
    # test_data = [('wut', 'mk2', .3)]
    # test_data = [('wut', 'x2', .1)]
    # test_data = [('fcps', 'wingnut', .5)]  # 1 / 0.71 -> Good
    # test_data = [('fcps', 'lsun', .1)]  # 1 / 0.35 -> Good
    # test_data = [('fcps', 'target', .1)]  # 1 / 0.5 -> Good
    test_data = [('graves', 'parabolic', .8)]  # 1 / 0.5
    # test_data = [('graves', 'zigzag', .25)]  # 1 / 0.5 -> Good
    for battery, dataset, theta_value in test_data:
        dataset = load_dataset(battery=battery, dataset=dataset)
        # plot_dataset(dataset)
        print('Starting Berserker')
        t_start = time.perf_counter()
        centroids, labels = berserker2(np.ascontiguousarray(dataset.data),
                                       theta_value, all_out=False, verbose=False)
        print('Duration is ', time.perf_counter() - t_start)
        print(f'cluster size: {len(centroids)} / {dataset.n_clusters} expected')
        results = DataSet(
            battery=dataset.battery,
            dataset=dataset.dataset,
            description=dataset.description,
            data=dataset.data,
            labels=[labels],
            n_clusters=[len(centroids)]
        )
        plot_dataset([results, dataset], title=['TB', 'Ground Truth'])
        print('Starting TB2')
        # import ipdb; ipdb.set_trace()
        centroids2, labels2 = berserker2(np.ascontiguousarray(centroids),
                                         1., all_out=False, verbose=False)
        print('Duration is ', time.perf_counter() - t_start)
        unq = np.unique(centroids2)
        hdb_nclusters = np.sum(unq >= 0)
        print(f'cluster size: {hdb_nclusters} / {dataset.n_clusters} expected')
        final_labels = labels.copy()
        for i, l in enumerate(labels):
            final_labels[i] = labels2[l]

        results = DataSet(
            battery=dataset.battery,
            dataset=dataset.dataset,
            description=dataset.description,
            data=dataset.data,
            labels=[final_labels],
            n_clusters=[hdb_nclusters]
        )
        # res = clustbench.load_results('*', dataset.battery, dataset.dataset,
        #                               results.n_clusters, path=RESULTS_PATH
        #                               )
        # labels2 = final_labels + 1
        # all_scores = {method: clustbench.get_score(labels2, res[method]) for method in res.keys()}
        # pprint(all_scores)
        nca = calculate_nca(dataset.labels[-1], labels)
        ri = rand_score(dataset.labels[-1], labels)
        fm = fowlkes_mallows_score(dataset.labels[-1], labels)
        ari = adjusted_rand_score_custom(dataset.labels[-1], labels)
        nmi = normalized_mutual_information(dataset.labels[-1], labels)

        print(f"Normalized Clustering Accuracy: {nca}")
        print(f"Rand Index (RI): {ri}")
        print(f"Fowlkes-Mallows Index (FM): {fm}")
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Normalized Mutual Information (NMI): {nmi}")

        plot_dataset([results, dataset], title=['TB-DBSCAN', 'Ground Truth'])


def test_sklearn_data():
    csv_table = [('Methods', 'Clusters', 'Runtime', 'Peak Memory', 'NCA', 'RI', 'FM', 'ARI', 'NMI')]
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    n_samples, n_features = X.shape

    np.random.seed(0)

    # ----------------------------------------------------------------------
    # Visualize the clustering
    def plot_clustering(X_red, labels, title=None):
        x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
        X_red = (X_red - x_min) / (x_max - x_min)

        plt.figure(figsize=(6, 4))
        for digit in digits.target_names:
            plt.scatter(
                *X_red[y == digit].T,
                marker=f"${digit}$",
                s=50,
                c=plt.cm.nipy_spectral(labels[y == digit] / 10),
                alpha=0.5,
            )

        plt.xticks([])
        plt.yticks([])
        if title is not None:
            plt.title(title, size=17)
        plt.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ----------------------------------------------------------------------
    # 2D embedding of the digits dataset
    print("Computing embedding")
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    print("Done.")

    plot_clustering(X_red, y, "Ground Truth")

    from sklearn.cluster import AgglomerativeClustering

    for linkage in ("tbscan", ):  # ("ward", "average", "complete", "single", "tb", "tbscan"):
        if linkage == "tb":
            theta_value = 0.00122
            tracemalloc.start()
            t_start = time.perf_counter()
            centroids, labels = berserker2(np.ascontiguousarray(X_red),
                                           theta_value, all_out=False, verbose=False)
            speed = time.perf_counter() - t_start
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem / (1024**2)
            tracemalloc.stop()
            print('Duration is ', speed)
            print(f'cluster size: {len(centroids)} / {len(np.unique(y))} expected')
            plot_clustering(X_red, labels, "TB")
            csv_data = ['TB', f'{len(centroids)}/{len(np.unique(y))}', f"{speed:.9f}", f"{peakmem:.4f}"]
        if linkage == "tbscan":
            theta_value = 0.00030
            eps = 0.00048
            min_samples = 1
            tracemalloc.start()
            t_start = time.perf_counter()
            centroids, labels = berserker2(np.ascontiguousarray(X_red),
                                           theta_value, all_out=False, verbose=False)
            print(f'cluster size: {len(centroids)} / {len(np.unique(y))} expected')
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(centroids)
            speed = time.perf_counter() - t_start
            unq = np.unique(model.labels_)
            hdb_nclusters = np.sum(unq >= 0)
            final_labels = labels.copy()
            for i, l in enumerate(labels):
                final_labels[i] = model.labels_[l]
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem / (1024**2)
            tracemalloc.stop()
            print('Duration is ', speed)
            print(f'cluster size: {hdb_nclusters} / {len(np.unique(y))} expected')
            plot_clustering(X_red, labels, "TB")
            plot_clustering(X_red, final_labels, "TBSCAN")
            csv_data = ['TBSCAN', f'{len(centroids)}/{len(np.unique(y))}', f"{speed:.9f}", f"{peakmem:.4f}"]
            labels = final_labels
        else:
            tracemalloc.start()
            t0 = time.perf_counter()
            clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
            clustering.fit(X_red)
            speed = time.perf_counter() - t0
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem = peakmem / (1024**2)
            tracemalloc.stop()
            print("%s :\t%.2fs" % (linkage, speed))

            plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)
            labels = clustering.labels_
            csv_data = [f'AC-{linkage}', f'{len(np.unique(labels))}/{len(np.unique(y))}', f"{speed:.9f}", f"{peakmem:.4f}"]

        nca = calculate_nca(y, labels)
        ri = rand_score(y, labels)
        fm = fowlkes_mallows_score(y, labels)
        ari = adjusted_rand_score_custom(y, labels)
        nmi = normalized_mutual_information(y, labels)

        print(f"Normalized Clustering Accuracy: {nca}")
        print(f"Rand Index (RI): {ri}")
        print(f"Fowlkes-Mallows Index (FM): {fm}")
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Normalized Mutual Information (NMI): {nmi}")

        csv_data.extend([f"{nca:.5f}", f"{ri:.5f}", f"{fm:.5f}", f"{ari:.5f}", f"{nmi:.5f}"])
        csv_table.append(csv_data)

    np.savetxt('hierarchical_results.csv', csv_table, delimiter=',', fmt='%s')
    plt.show()


def test_comparison_algo():
    linear_methods = ['sklearn_kmeans', 'sklearn_birch_T0.005_BF10', 'sklearn_gm',]
    non_linear_methods = ['spectral' ]  #  'HDBSCAN', 'hierarchical', 'FCPS_HDBSCAN_2', 'sklearn_spectral_Arbf_G0.25', 'sklearn_spectral_Arbf_G0.5', 'sklearn_spectral_Arbf_G1', 'sklearn_spectral_Arbf_G2.5', 'sklearn_spectral_Arbf_G5', 'sklearn_spectral_Alaplacian_G0.25', 'sklearn_spectral_Alaplacian_G0.5', 'sklearn_spectral_Alaplacian_G1', 'sklearn_spectral_Alaplacian_G2.5', 'sklearn_spectral_Alaplacian_G5', 'sklearn_spectral_Apoly_G0.25', 'sklearn_spectral_Apoly_G0.5', 'sklearn_spectral_Apoly_G1', 'sklearn_spectral_Apoly_G2.5', 'sklearn_spectral_Apoly_G5', 'sklearn_spectral_Asigmoid_G0.25', 'sklearn_spectral_Asigmoid_G0.5', 'sklearn_spectral_Asigmoid_G1', 'sklearn_spectral_Asigmoid_G2.5', 'sklearn_spectral_Asigmoid_G5'
    linear_data = [('wut', 'circles', 4), ('wut', 'z3', 4), ('sipu', 'birch1', 100),
                   ('wut', 'mk1', 3), ('wut', 'trajectories', 4), ('wut', 'x1', 3),
                   ('sipu', 'a1', 20), ('sipu', 'r15', 8)]
    non_linear_data = [('wut', 'windows', 5), ('other', 'square', 2),
                       ('fcps', 'wingnut', 2), ('graves', 'zigzag', 3),
                       ('fcps', 'target', 6), ('fcps', 'lsun', 3),
                       ('wut', 'mk2', 2)
                       ]

    datas = non_linear_data  # linear_data
    methods = non_linear_methods  # linear_methods

    for method in methods:
        csv_table = [('Dataset', 'Final N_Clusters', 'Speed', 'NCA', 'RI', 'FM', 'ARI', 'NMI')]
        print(f"Method: {method}")
        # print(f"Score: {clustbench.get_score(c_dataset.labels[-1], res[method])}")
        for battery, dataset, nb_cluster in datas:
            csv_data = [dataset, f"{nb_cluster}/{nb_cluster}"]

            c_dataset = clustbench.load_dataset(battery, dataset, path=DATA_PATH)
            if method == 'hierarchical':
                linkage = 'single'  # "ward", "average", "complete", "single"
                tracemalloc.start()
                t0 = time.perf_counter()
                clustering = AgglomerativeClustering(linkage=linkage, n_clusters=nb_cluster)
                clustering.fit(c_dataset.data)
                speed = time.perf_counter() - t0
                amem, peakmem = tracemalloc.get_traced_memory()
                peakmem = peakmem / (1024**2)
                tracemalloc.stop()
                print("%s :\t%.2fs" % (linkage, speed))

                labels = clustering.labels_
                csv_data.append(f"{speed:.9f}")
            if method == 'spectral':
                tracemalloc.start()
                t0 = time.perf_counter()
                clustering = SpectralClustering(n_clusters=nb_cluster,)
                clustering.fit(c_dataset.data)
                speed = time.perf_counter() - t0
                amem, peakmem = tracemalloc.get_traced_memory()
                peakmem = peakmem / (1024**2)
                tracemalloc.stop()

                labels = clustering.labels_
                csv_data.append(f"{speed:.9f}")
            elif method == 'HDBSCAN':
                t_start = time.perf_counter()
                model = HDBSCAN(min_samples=1,
                                min_cluster_size=nb_cluster,
                                cluster_selection_epsilon=3.0)
                model.fit(c_dataset.data)
                speed = time.perf_counter() - t_start

                labels = model.labels_
                csv_data.append(f"{speed:.9f}")

            else:
                csv_data.append('')
                # o_dataset = load_dataset(battery=battery, dataset=dataset)
                res = clustbench.load_results('*', battery, dataset, nb_cluster,
                                            path=RESULTS_PATH)
                # print(res.keys())
                print(f"Results for {battery}/{dataset} with {nb_cluster} clusters:")

                labels = res.get(method, {}).get(nb_cluster, None)
                if labels is None:
                    print(f"Method {method} not found or labels not found")
                    print(res.keys())
                    csv_data.extend(['', '', '', '', ''])
                    csv_table.append(csv_data)
                    continue
            nb_clusters = len(np.unique(labels))
            results = DataSet(
                battery=c_dataset.battery,
                dataset=c_dataset.dataset,
                description=c_dataset.description,
                data=c_dataset.data,
                labels=[labels],
                n_clusters=[nb_clusters]
            )
            plot_dataset([results, c_dataset], title=[f'{method}', 'Ground Truth'])

            nca = calculate_nca(c_dataset.labels[0], labels)
            ri = rand_score(c_dataset.labels[0], labels)
            fm = fowlkes_mallows_score(c_dataset.labels[0], labels)
            ari = adjusted_rand_score_custom(c_dataset.labels[0], labels)
            nmi = normalized_mutual_information(c_dataset.labels[0], labels)

            print(f"Normalized Clustering Accuracy: {nca}")
            print(f"Rand Index (RI): {ri}")
            print(f"Fowlkes-Mallows Index (FM): {fm}")
            print(f"Adjusted Rand Index (ARI): {ari}")
            print(f"Normalized Mutual Information (NMI): {nmi}")

            csv_data.extend([f"{nca:.5f}", f"{ri:.5f}", f"{fm:.5f}", f"{ari:.5f}", f"{nmi:.5f}"])
            csv_table.append(csv_data)

        np.savetxt(f'{method}_results.csv', csv_table, delimiter=',', fmt='%s')


if __name__ == "__main__":
    # print(get_battery_names())
    # print(get_dataset_names(battery_name='mnist'))
    # test_tb()
    # test_tb_hdbscan()
    # test_tb_tb()
    # test_sklearn_data()
    test_comparison_algo()
