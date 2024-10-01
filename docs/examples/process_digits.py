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
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics import pair_confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from thetan.berserker import berserker2



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

    for linkage in ("ward", "average", "complete", "single", "tb"):
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
    # plt.show()
