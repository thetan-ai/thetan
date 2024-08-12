# This code uses dataset S1 to S4
# The datasets can be downloaded here
# https://cs.joensuu.fi/sipu/datasets/

import os
from time import time

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score

from thetan.berserker import berserker2


test_params = {'s1':[0.08, 0.58],
               's2':[0.05, 0.62],
               's3':[0.05, 0.68],
               's4':[0.05, 0.78]}

metric = "euclidean"
dataset_dir = 'S-sets'
n_clusters = 15

datasets = [k for k in test_params.keys()]
for i in range(4):
    times = []
    ARSs = []
    NMIs = []
    FMSs = []
    dataset = datasets[i]

    X = []
    with open(os.path.join(dataset_dir, dataset+'.txt')) as file:
        for line in file:
            X.append([int(n) for n in line.split()])

    X = np.ascontiguousarray(X)
    print(X.shape)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    for iter_n in range(10):
        s_t = time()
        try:
            centroids, labels_ = berserker2(X, test_params[dataset][1], metric='euclidean', all_out=False, verbose=False)
            uniq, sizes = np.unique(labels_, return_counts=True)
            
            biggest_k = np.argsort(sizes)[-n_clusters:]
            centroids_tmp = np.ascontiguousarray(centroids[biggest_k].copy())

            kmeans = KMeans(n_clusters=n_clusters, init=centroids_tmp, n_init=1,
                            max_iter=300, tol=0.0001).fit(X)
        except:
            iter_n -= 1
            continue
        
        times.append(time()-s_t)
        
        kcentroids = kmeans.cluster_centers_
        klabels = kmeans.labels_

        n_clusters_ = 15
        labels_ = []
        with open(os.path.join(dataset_dir, dataset+"-label.txt")) as file:
            for idx, line in enumerate(file):
                labels_.append(int(line)-1)
        
        gt_labels = np.array(labels_)
        
        ars = adjusted_rand_score(gt_labels, klabels)
        ARSs.append(ars)
        nmi = normalized_mutual_info_score(gt_labels, klabels)
        NMIs.append(nmi)
        fms = fowlkes_mallows_score(gt_labels, klabels)
        FMSs.append(fms)
    print('ARS: ', np.mean(ARSs), u"\u00B1", np.std(ARSs))
    print('NMI: ', np.mean(NMIs), u"\u00B1", np.std(NMIs))
    print('FMS: ', np.mean(FMSs), u"\u00B1", np.std(FMSs))
    print('TIME: ', np.mean(times), u"\u00B1", np.std(times))