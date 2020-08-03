''' Fast Chaing Node Method'''

import numpy as np
from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import IdentityFeature
from dipy.segment.metric import EuclideanMetric
import time
from fury import actor, window


def dist(a, b):

    #  np.sqrt(np.sum((a-b)**2))
    return np.linalg.norm(a-b)


def fcn(data, qbx_thresholds, chaining_thr, viz):
    '''Fast Chaining Node

    Parameters
    ----------
    data : sequence of features
        input data to be clustered
    qbx_thresholds: sequence
        ist of distance thresholds for for QBX
    chaining_thr: float
        chaining threshold
    viz: bool
        visualize the result if true
    '''

    # We first get initial clusters using QBX

    vector_feature = IdentityFeature()
    metric = EuclideanMetric(feature=vector_feature)

    qbx = QuickBundlesX(thresholds=qbx_thresholds, metric=metric)

    clusters = qbx.cluster(data)

    # get clusters of layer 1 (we have only one layer)
    total_clusters = clusters.get_clusters(1)
    # centroids of to clusters
    total_centroids = np.array(total_clusters.centroids)
    # reshape data for visualization
    data = np.array(data)
    data = data.reshape(data.shape[0], 3)
    total_centroids = total_centroids.reshape(total_centroids.shape[0], 3)

    all_centroids = total_centroids.copy()

    # array that has cluster numbers for each data point
    cluster_no = np.zeros(len(data))
    # an array that has cluster numbers for each cluster centroid
    c_cluster_no = np.zeros(len(total_clusters))

    for i in range(len(total_clusters)):
        cluster_no[total_clusters[i]] = i
        '''
        assign cluster number to the points that belong in that cluster
        0th index means 0th point in data
        index tells the data point number
        value on that index means cluster number it belongs to
        '''
        c_cluster_no[i] = i  # same goes for cluster centroid numbers

    print("total number of initial clusters = ", len(list(set(cluster_no))))

    # visualize the clusters
    if viz:

        ren = window.Renderer()
        m = int(max(set(cluster_no)) + 1)

        colors = [np.random.rand(3) for streamline in range(m)]

        p_colors = [[[0, 0, 0]]]
        for i in range(len(data)):
            p_colors.append(colors[int(cluster_no[i])])

        p_colors.pop(0)

        ren.background((1, 1, 1))
        ren.add(actor.sphere(data, colors=p_colors, radii=0.25))

        window.show(ren)

    ''' Cluster chaining and re-labelling data/clusters part '''

    f_start_time = time.time()

    # chaining clusters

    '''
    cluster_chaining is list of lists
    each list on ith index corresponds to the ith data point
    ith list will have all the cluster numbers ith data point belongs to

    '''
    iteration = 10  # number of iterations you want to run
    total_clusters = list(set(cluster_no))
    theta = chaining_thr
    old_clusters = total_clusters.copy()
    for ite in range(iteration):

        cluster_chaining = [[1]]
        for p in range(len(c_cluster_no)):  # all_centroids
            # initially just assigning it the original
            # cluster numbers we got from qbx
            cluster_chaining.append([c_cluster_no[p]])
        cluster_chaining.pop(0)

        for ck in range(len(total_clusters)):  # cluster number

            cc = c_cluster_no[ck]
            ct = all_centroids[ck]
            # should look at new centroids for this cluster too
            remaining_cts = np.argwhere(c_cluster_no != cc)

            for k in remaining_cts:
                # k has point number that belongs to complement of cluster ck
                k = k[0]
                pointy = all_centroids[k]  # point value

                if dist(ct, pointy) <= theta:
                    # if within threshold, add to cluster
                    cluster_chaining[k].append(cc)

        relabled = cluster_chaining.copy()
        # relabel it to the smallest cluster number
        for r in range(len(cluster_chaining)):

            relabled[r] = min(cluster_chaining[r])

            other_points = cluster_chaining[r]

            for j in other_points:
                # j has point number that belongs to cluster ck
                if j != relabled[r]:
                    remaining_points = np.argwhere(cluster_no == j)
                    for k in remaining_points:
                        # k has point number that belongs to
                        # complement of cluster ck
                        k = k[0]
                        cluster_no[k] = relabled[r]

                    remaining_points = np.argwhere(c_cluster_no == j)
                    for k in remaining_points:
                        # k has point number that belongs to complement
                        # of cluster ck
                        k = k[0]
                        c_cluster_no[k] = relabled[r]

                        # edit
                        cluster_chaining[k].append(relabled[r])

        # total_clusters = list(set(relabled)) # number of clusters left.
        # like cluster 4, 7 9 ,10

        new_clusters = len(list(set(cluster_no)))

        print("total number of clusters = ", new_clusters)

        if old_clusters == new_clusters:
            print("reached convergence")
            break
        old_clusters = new_clusters

    print("--- %s seconds ---" % (time.time() - f_start_time))

    print("total number of final clusters = ", len(list(set(cluster_no))))

    if viz:

        or_p = list(set(cluster_no))
        pc = list(range(len(set(cluster_no))))

        ren = window.Renderer()
        # visualize final result
        m = int(len(set(cluster_no)) + 1)

        colors = [np.random.rand(3) for streamline in range(m)]

        p_colors = [[[0, 0, 0]]]
        for i in range(len(data)):

            p_colors.append(colors[pc[np.where(cluster_no[i] == or_p)[0][0]]])

        p_colors.pop(0)

        ren.background((1, 1, 1))
        ren.add(actor.sphere(data, colors=p_colors, radii=0.35))
        window.show(ren)
