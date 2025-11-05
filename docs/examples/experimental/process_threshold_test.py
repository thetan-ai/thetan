# This code uses dataset S1 to S4
# The datasets can be downloaded here
# https://cs.joensuu.fi/sipu/datasets/

import os
from time import time


import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler

from thetan.berserker import berserker2
from pyclustering.cluster.bsas import bsas


datasets = ['s1', 's2', 's3', 's4']
dataset_dir = 'S-sets'

for i in range(4):
    dataset = datasets[i]
    X = []
    with open(os.path.join(dataset_dir, dataset+'.txt')) as file:
        for line in file:
            X.append([int(n) for n in line.split()])

    X = np.ascontiguousarray(X)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    times = []
    n_clusters = []

    for th in np.arange(0.1, 1.02, 0.02):
        s_t = time()
        results = berserker2(X, th, metric='euclidean', all_out=False)
        times.append(time()-s_t)
        n_clusters_ = len(results[0])
        n_clusters.append(n_clusters_)
    
    if os.path.exists('threshold_plots') == False:
        os.makedirs('threshold_plots')

    plt.figure()
    plt.plot(np.arange(0.1, 1.02, 0.02), n_clusters)
    plt.xlabel('Threshold value')
    plt.ylabel('Number of clusters')
    plt.savefig(os.path.join('threshold_plots', dataset+'_tb.png'))
    plt.close()

    plt.figure()
    plt.plot(np.arange(0.1, 1.02, 0.02), times)
    plt.xlabel('Threshold value')
    plt.ylabel('Time taken (sec)')
    plt.savefig(os.path.join('threshold_plots', dataset+'_tb_times.png'))
    plt.close()

    tb_res = n_clusters

    times = []
    n_clusters = []

    for th in np.arange(0.1, 1.02, 0.02):
        s_t = time()
        shuffled_X = X.copy()
        np.random.shuffle(shuffled_X)
        bsas_instance = bsas(shuffled_X, 1000, th)
        bsas_instance.process()
        times.append(time()-s_t)
        n_clusters_ = len(bsas_instance.get_clusters())
        n_clusters.append(n_clusters_)

    if os.path.exists('threshold_plots') == False:
        os.makedirs('threshold_plots')

    plt.figure()
    plt.plot(np.arange(0.1, 1.02, 0.02), n_clusters)
    plt.xlabel('Threshold value')
    plt.ylabel('Number of clusters')
    plt.savefig(os.path.join('threshold_plots', dataset+'_bsas.png'))
    plt.close()

    plt.figure()
    plt.plot(np.arange(0.1, 1.02, 0.02), times)
    plt.xlabel('Threshold value')
    plt.ylabel('Time taken (sec)')
    plt.savefig(os.path.join('threshold_plots', dataset+'_bsas_times.png'))
    plt.close()

    bsas_res = n_clusters

    plt.figure()
    plt.title(dataset)
    plt.plot(np.arange(0.1, 1.02, 0.02), bsas_res, label='BSAS')
    plt.plot(np.arange(0.1, 1.02, 0.02), tb_res, label='TB (Ours)')
    plt.xlabel('Threshold value')
    plt.ylabel('Number of clusters')
    plt.legend()
    plt.savefig(os.path.join('threshold_plots', dataset+'_full.png'))
    plt.close()