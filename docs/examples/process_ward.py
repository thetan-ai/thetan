import numpy as np
from time import time
from sklearn import datasets, manifold
import matplotlib.pyplot as plt
import tracemalloc
from dataclasses import make_dataclass
import os
from os.path import join as pjoin
from pprint import pprint
import tracemalloc
import time
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment
from sklearn import datasets, manifold
from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pair_confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from thetan.berserker import berserker2, berserker2_clean
from thetan.theta import centroids_from_labels, assign


def plot_centroids_clusters(data, centroids, labels=None, title=None, exp_type=None, gt_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True, maxwindow=False, savefn=None, centroids_label=True,
                            digits=None):

    new_colors = np.zeros((len(data), 3))

    if centroids is not None:
        no_clusters = len(centroids)
        cluster_colors = np.random.rand(no_clusters, 3)
    else:
        if labels is not None:
            no_clusters = np.max(labels) + 1
            cluster_colors = np.random.rand(no_clusters, 3)

    if labels is not None:

        for ind in range(len(data)):
            new_colors[ind] = cluster_colors[labels[ind]]

    fig = plt.figure()
    plt.title(title)
    plt.axis('equal')
    # plt.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)

    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(labels[i]), c=new_colors[i], fontsize=8)

    if digits is not None:
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(digits[i]), c=new_colors[i], fontsize=8)

    # if digits is not None:
    # for digit in digits.target_names:
    #         plt.scatter(
    #             *X_red[y == digit].T,
    #             marker=f"${digit}$",
    #             s=50,
    #             c=plt.cm.nipy_spectral(labels[y == digit] / 10),
    #             alpha=0.5,
    #         )


    if centroids is not None:

        plt.plot(centroids[:, 0], centroids[:, 1], 'rx')

        if centroids_label:
            for c in range(centroids.shape[0]):
                plt.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if gt_centroids is not None:
        plt.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'bx')

    if uncertain_data is not None:
        plt.plot(uncertain_data[:, 0], uncertain_data[:, 1], 'g.')

    cnt = 0
    if final_counts is not None:
        for i in range(final_counts.shape[0]):
            prev = cnt
            cnt += final_counts[i]
            if final_counts[i] > 1:
                plt.text(data[i, 0], data[i, 1], str(final_nearc_value[prev: cnt]) + ' ' + str(labels[i]), fontsize=8)

    if merged_centroids is not None:
        plt.plot(merged_centroids[:, 0], merged_centroids[:, 1], 'mo')

    # pl.get_current_fig_manager().full_screen_toggle()

    if maxwindow:
        plt.get_current_fig_manager().window.showMaximized()

    if show:
        plt.show()

    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight')


def tbscan(X, theta_value, eps=2., min_samples=1, return_interm=False):

    # theta_value = 3.6

    # eps = 2.
    # min_samples = 1

    initial_centroids, initial_labels = berserker2_clean(X, theta_value)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(initial_centroids)

    unq = np.unique(model.labels_)
    db_nclusters = np.sum(unq>=0)

    dbcentroids = np.array([np.mean(initial_centroids[model.labels_==l], axis=0) for l in range(db_nclusters)])
    dblabels = model.labels_

    final_labels = initial_labels.copy()

    for i, l in enumerate(initial_labels):
        final_labels[i] = dblabels[l]

    centroids = dbcentroids
    labels = final_labels

    if return_interm:
        return centroids, labels, initial_centroids, initial_labels
    else:
        return centroids, labels


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



    for linkage in ("ward", "average", "complete", "single", "tb", "tbscan"):
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

        elif linkage == "tbscan":
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

        print()
        print('Algorithm', linkage)
        print(f"Normalized Clustering Accuracy: {nca}")
        print(f"Rand Index (RI): {ri}")
        print(f"Fowlkes-Mallows Index (FM): {fm}")
        print(f"Adjusted Rand Index (ARI): {ari}")
        print(f"Normalized Mutual Information (NMI): {nmi}")
        print()

        csv_data.extend([f"{nca:.5f}", f"{ri:.5f}", f"{fm:.5f}", f"{ari:.5f}", f"{nmi:.5f}"])
        csv_table.append(csv_data)

    print()
    print(csv_table)
    print()

    np.savetxt('hierarchical_results_new.csv', csv_table, delimiter=',', fmt='%s')
    plt.show()


def analysis_with_tb_max_nca():

    print('Load data')
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    print("Computing embedding")
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    X_red = np.ascontiguousarray(X_red)
    centroids = centroids_from_labels(X_red, np.ascontiguousarray(y, dtype='i8'))

    theta_value = 0.00122
    # theta_value = 0.00030

    for theta_value in np.linspace(0.0011, 0.0012424, 500):

        tb_centroids, tb_labels = berserker2(X_red,
                                             theta_value, all_out=False, verbose=False)

        labels = tb_labels

        nca = calculate_nca(y, labels)
        ri = rand_score(y, labels)
        fm = fowlkes_mallows_score(y, labels)
        ari = adjusted_rand_score_custom(y, labels)
        nmi = normalized_mutual_information(y, labels)

        if (nca > 0.57):
            print()
            print('Theta ', theta_value)
            print(tb_centroids.shape)
            print(f"Normalized Clustering Accuracy: {nca}")
            print(f"Rand Index (RI): {ri}")
            print(f"Fowlkes-Mallows Index (FM): {fm}")
            print(f"Adjusted Rand Index (ARI): {ari}")
            print(f"Normalized Mutual Information (NMI): {nmi}")
            print()

    # plot_centroids_clusters(X_red, centroids, y, 'Ground Truth', show=False)

    # plot_centroids_clusters(X_red, tb_centroids, tb_labels, 'TB Output at value '  + str(theta_value), show=False, digits=y)

    # plt.show()


    tb_uniq, tb_sizes = np.unique(tb_labels, return_counts=True)


def analysis_with_tbscan():

    print('Load data')
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    print("Computing embedding")
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
    X_red = np.ascontiguousarray(X_red)
    centroids = centroids_from_labels(X_red, np.ascontiguousarray(y, dtype='i8'))

    # theta_value = 0.00122
    # theta_value = 0.00030
    theta_value = 0.00010
    eps = 1.5 * theta_value
    min_samples = 3


    # for eps in np.linspace(1.8 * theta_value, 2.5 * theta_value, 10):
    #     for min_samples in [1, 2, 3, 4, 5, 6]:

    tbscan_centroids, tbscan_labels, tb_centroids, tb_labels = tbscan(X_red, theta_value, eps, min_samples, return_interm=True)

    if tbscan_centroids.shape[0] > 5:

        print()
        print('TBSCAN params', theta_value, eps, min_samples)
        print(tbscan_centroids.shape)

        labels = tbscan_labels

        nca = calculate_nca(y, labels)
        ri = rand_score(y, labels)
        fm = fowlkes_mallows_score(y, labels)
        ari = adjusted_rand_score_custom(y, labels)
        nmi = normalized_mutual_information(y, labels)

        if nca > 0.5:


            print(f"Normalized Clustering Accuracy: {nca}")
            print(f"Rand Index (RI): {ri}")
            print(f"Fowlkes-Mallows Index (FM): {fm}")
            print(f"Adjusted Rand Index (ARI): {ari}")
            print(f"Normalized Mutual Information (NMI): {nmi}")
            print()


    plot_centroids_clusters(X_red, centroids, y, 'Ground Truth', show=False, digits=y)

    plot_centroids_clusters(X_red, tb_centroids, tb_labels, 'TB Output at value '  + str(theta_value), show=False, digits=y)

    plot_centroids_clusters(X_red, tbscan_centroids, tbscan_labels, 'TBSCAN Output at value '  + str(theta_value), show=False, digits=y)

    plt.show()

    tb_uniq, tbscan_sizes = np.unique(tbscan_labels, return_counts=True)

    print(sorted(tbscan_sizes)[::-1][:20])


    t = 1



def analysis_with_hc(linkage, n_clusters, ndims=2, reassign=False):

    # print('Load data')
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    print(X.shape)

    if ndims == 64:
        X_red = X
    else:
        # print("Computing embedding")
        X_red = manifold.SpectralEmbedding(n_components=ndims).fit_transform(X)

    X_red = np.ascontiguousarray(X_red)
    print(X_red.shape)

    centroids = centroids_from_labels(X_red, np.ascontiguousarray(y, dtype='i8'))

    start = time.perf_counter()
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(X_red)
    duration = time.perf_counter() - start
    print('Duration ', duration)

    tb_labels = clustering.labels_

    tb_centroids = centroids_from_labels(X_red, tb_labels)

    if reassign:

        tb_uniq, tb_sizes = np.unique(tb_labels, return_counts=True)
        tb_centroids2 = tb_centroids[np.argsort(tb_sizes)[::-1][:10]]

        tb_labels2 = assign(X_red, tb_centroids2)

        labels2 = tb_labels2

    else:

        labels2 = tb_labels

    nca = calculate_nca(y, labels2)
    ri = rand_score(y, labels2)
    fm = fowlkes_mallows_score(y, labels2)
    ari = adjusted_rand_score_custom(y, labels2)
    nmi = normalized_mutual_information(y, labels2)

    # if ((nca > 0.67) & (nmi > 0.63)):
    print()
    print(linkage)
    # print(tb_centroids.shape, ' to ', tb_centroids2.shape)
    print(f"Normalized Clustering Accuracy: {nca}")
    print(f"Rand Index (RI): {ri}")
    print(f"Fowlkes-Mallows Index (FM): {fm}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print()

    # plot_centroids_clusters(X_red, centroids, y, 'Ground Truth', show=False, digits=y)

    # plot_centroids_clusters(X_red, tb_centroids2, tb_labels2, linkage + ' Output at value '  + str(n_clusters), show=False, digits=y)

    # plt.show()

    return duration, -1, nca, ri, fm, ari, nmi


def analysis_with_tb(theta_value = 0.00112, ndims=2):

    # print('Load data')
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    print(X.shape)


    # print("Computing embedding")

    if ndims == 64:
        X_red = X
    else:
        X_red = manifold.SpectralEmbedding(n_components=ndims).fit_transform(X)

    X_red = np.ascontiguousarray(X_red)
    print(X_red.shape)

    centroids = centroids_from_labels(X_red, np.ascontiguousarray(y, dtype='i8'))

    # theta_value = 0.00112
    t_start = time.perf_counter()
    tb_centroids, tb_labels = berserker2_clean(X_red,
                                               theta_value)
    duration = time.perf_counter() - t_start
    print('Duration ', duration)

    labels = tb_labels

    nca = calculate_nca(y, labels)
    ri = rand_score(y, labels)
    fm = fowlkes_mallows_score(y, labels)
    ari = adjusted_rand_score_custom(y, labels)
    nmi = normalized_mutual_information(y, labels)


    # print()
    # print('Theta ', theta_value)
    # print(tb_centroids.shape)
    # print(f"Normalized Clustering Accuracy: {nca}")
    # print(f"Rand Index (RI): {ri}")
    # print(f"Fowlkes-Mallows Index (FM): {fm}")
    # print(f"Adjusted Rand Index (ARI): {ari}")
    # print(f"Normalized Mutual Information (NMI): {nmi}")
    # print()

    # plot_centroids_clusters(X_red, centroids, y, 'Ground Truth', show=False)

    # plot_centroids_clusters(X_red, tb_centroids, tb_labels, 'TB Output at value '  + str(theta_value), show=False, digits=y)

    # plt.show()


    tb_uniq, tb_sizes = np.unique(tb_labels, return_counts=True)
    tb_centroids2 = tb_centroids[np.argsort(tb_sizes)[::-1][:10]]

    tb_labels2 = assign(X_red, tb_centroids2)

    labels2 = tb_labels2

    nca = calculate_nca(y, labels2)
    ri = rand_score(y, labels2)
    fm = fowlkes_mallows_score(y, labels2)
    ari = adjusted_rand_score_custom(y, labels2)
    nmi = normalized_mutual_information(y, labels2)

    # if ((nca > 0.70) & (nmi > 0.63)):
    print()
    print('Assign ', theta_value)
    print(tb_centroids.shape, ' to ', tb_centroids2.shape)
    print(f"Normalized Clustering Accuracy: {nca}")
    print(f"Rand Index (RI): {ri}")
    print(f"Fowlkes-Mallows Index (FM): {fm}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print()

    # plot_centroids_clusters(X_red, centroids, y, 'Ground Truth', show=False, digits=y)

    # plot_centroids_clusters(X_red, tb_centroids2, tb_labels2, 'TB Output at value '  + str(theta_value), show=False, digits=y)

    # plt.show()

    return duration, -1, nca, ri, fm, ari, nmi



def collect_statistics_ndims2():

    res_ward = analysis_with_hc('ward', n_clusters=10)
    res_average = analysis_with_hc('average', n_clusters=10)
    res_complete = analysis_with_hc('complete', n_clusters=10)
    res_single = analysis_with_hc('single', n_clusters=10)


    # analysis_with_ward('ward', n_clusters=45)
    # for i in range(20, 60, 2):
    #     analysis_with_hc('ward', n_clusters=i)

    # for i in range(100):
    #     for factor in np.linspace(0.38, 0.48, 100):
    #     # for factor in [0.8, 0.75, 0.7, 0.65, 0.6]:
    #     # for factor in [0.6, 0.5, 0.45, 0.4]:
    #         print(">> Factor ", factor)
    #         analysis_with_tb(theta_value = factor * 0.00112)

    res_tb = analysis_with_tb(theta_value = 0.00048)

    A = np.zeros((5, 7))
    A[0] = res_ward
    A[1] = res_average
    A[2] = res_complete
    A[3] = res_single
    A[4] = res_tb

    import pandas as pd

    # Define the row and column names
    rows = ["Ward", "Average", "Complete", "Single", "TB"]
    columns = ["Runtime", "Peak Memory", "NCA", "RI", "FM", "ARI", "NMI"]

    # Create a DataFrame with empty values
    df = pd.DataFrame(index=rows, columns=columns)

    df[:] = A

    # Display the DataFrame
    print(df)

    print()
    print()

    print(df.to_latex())


if __name__ == "__main__":

    # print('>>>>>>>>>>>>>>>>>>>>>')
    # analysis_with_tb(np.sqrt(4) * 0.00112, ndims=4)

    # print('>>>>>>>>>>>>>>>>>>>>>')
    # analysis_with_tb(np.sqrt(4) * 0.00048, ndims=4)

    print('>>>>>>>>>>>>>>>>>>>>>')

    # for i, v in enumerate(np.linspace(0.0036, 0.0056, 60)):
    #     print(i, v)
    #     analysis_with_tb(v, ndims=10)

    # analysis_with_tb(0.005566101, ndims=10)
    # analysis_with_hc('ward', 10, ndims=10)

    for i, v in enumerate(np.linspace(25, 35, 100)):
        print()
        print(i, v)
        res = analysis_with_tb(v, ndims=64)

        duration, _, nca, ri, fm, ari, nmi = res

        if nca > 0.82:
            print('........................>>>>>>>>>>>>>>>>>>>')

    # analysis_with_hc('ward', 10, ndims=64)
    # analysis_with_hc('average', 10, ndims=64)
    # analysis_with_hc('complete', 10, ndims=64)



    # ranges = 1.6 * np.linspace(0.0036, 0.0056, 60)
    # for i, v in enumerate(ranges):
    #     print(i, v)
    #     analysis_with_tb(v, ndims=20)