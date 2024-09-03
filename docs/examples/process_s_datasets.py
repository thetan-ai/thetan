# This code uses dataset S1 to S4
# The datasets can be downloaded here
# https://cs.joensuu.fi/sipu/datasets/

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler

from thetan.berserker import berserker2


test_params = {'s1':[0.08, 0.6],
               's2':[0.05, 0.64],
               's3':[0.05, 0.7],
               's4':[0.05, 0.8]}
colors = ["#ffb300", "#803e75", "#ff6800", "#a6bdd7",
          "#c10020", "#cea262", "#817066", "#007d34",
          "#f6768e", "#00538a", "#ff7a5c", "#53377a",
          "#ff8e00", "#b32851", "#f4c800", "#7f180d",
          "#93aa00", "#593315", "#f13a13", "#232c16"]
# Plot clustering results
metric = "euclidean"
n_clusters = 15

dataset_dir = 'S-sets'
output_dir = ''

datasets = [k for k in test_params.keys()]
methods = ['agglo', 'bk', 'kmp', 'km', 'dbscan', 'hdbscan', 'berserker']
for i in range(4):
    times = []
    dataset = datasets[i]
    dataset_output_path = os.path.join(output_dir, dataset)
    if os.path.exists(dataset_output_path) == False:
        os.makedirs(dataset_output_path)

    X = []
    with open(os.path.join(dataset_dir, dataset+'.txt')) as file:
        for line in file:
            X.append([int(n) for n in line.split()])

    X = np.ascontiguousarray(X)
    print(X.shape)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    s_t = time()
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average", metric=metric
    )
    model.fit(X)
    times.append(s_t-time())
    plt.figure()
    plt.title("AgglomerativeClustering")
    for l, color in zip(np.arange(model.n_clusters), colors):
        indices = X[model.labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'agglo.png'))

    s_t = time()
    model = BisectingKMeans(
        n_clusters=n_clusters, n_init=1)
    model.fit(X)
    times.append(s_t-time())
    plt.figure()
    plt.title("BisectingKMeans")
    for l, color in zip(np.arange(model.n_clusters), colors):
        indices = X[model.labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'bkmeans.png'))

    s_t = time()
    model = KMeans(
        n_clusters=n_clusters, n_init='auto')
    model.fit(X)
    times.append(s_t-time())
    plt.figure()
    plt.title("KMeans")
    for l, color in zip(np.arange(model.n_clusters), colors):
        indices = X[model.labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'kmeansp.png'))

    s_t = time()
    model = KMeans(
        n_clusters=n_clusters, n_init='auto', init='random')
    model.fit(X)
    times.append(s_t-time())
    plt.figure()
    plt.title("KMeans")
    for l, color in zip(np.arange(model.n_clusters), colors):
        indices = X[model.labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'kmeans.png'))

    s_t = time()
    model = DBSCAN(eps=test_params[dataset][0], min_samples=3)
    model.fit(X)
    times.append(s_t-time())
    labels_ = model.labels_
    n_clusters_ = len(set(labels_)) - (1 if -1 in labels_ else 0)
    plt.figure()
    plt.title("DBSCAN")
    for l, color in zip(np.arange(n_clusters), colors):
        indices = X[labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'dbscan.png'))

    s_t = time()
    model = HDBSCAN()
    model.fit(X)
    times.append(s_t-time())
    labels_ = model.labels_
    n_clusters_ = len(set(labels_)) - (1 if -1 in labels_ else 0)
    plt.figure()
    plt.title("HDBSCAN")
    for l, color in zip(np.arange(n_clusters), colors):
        indices = X[labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'hdbscan.png'))

    s_t = time()
    results = berserker2(X, test_params[dataset][1], metric='euclidean', all_out=False)
    times.append(s_t-time())
    labels_ = results[1]
    n_clusters_ = len(results[0])
    plt.figure()
    plt.title("TB")
    for l, color in zip(np.arange(n_clusters_), colors):
        indices = X[labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'TB.png'))


    n_clusters_ = 15
    labels_ = []
    with open(os.path.join(dataset_dir, dataset+"-label.txt")) as file:
        for idx, line in enumerate(file):
            labels_.append(int(line))

    plt.figure()
    plt.title("ground truth")
    for l, color in zip(np.arange(1, n_clusters_+1), colors):
        indices = X[labels_ == l]
        plt.scatter(indices[..., 0], indices[..., 1], c=color, alpha=0.5)

    plt.savefig(os.path.join(dataset_output_path, 'gt.png'))

    plt.bar(methods, times, color=colors[:len(methods)])
    plt.xticks(fontsize=8)
    plt.xlabel('methods')
    plt.ylabel('time (sec)')

    plt.savefig(os.path.join(dataset_output_path, 'time.png'))