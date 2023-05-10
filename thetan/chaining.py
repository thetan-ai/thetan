import time
import numpy as np


def dist(a, b):

    return np.linalg.norm(a-b)


def clusters_from_labels(centroids, labels):

    K = centroids.shape[0]

    clusters = [[] for k in range(K)]
    # alternatively for dictionary output
    # clusters = {k:[] for k in range(5)}

    for i, l in enumerate(labels):
        clusters[l].append(i)

    clusters_sizes = [len(c) for c in clusters]
    # or alternatively use np.unique
    # unique_labels, clusters_sizes = np.unique(labels, return_counts=True)

    return clusters, clusters_sizes



def chainer2(X, centroids, chaining_thr, iterations=1, clusters_sizes=None):
    """ Very experimental
    """

    K = centroids.shape[0]

    chains = [[] for k in range(K)]
    for c_index in range(K):
        for c_index2 in range(K):
            if c_index != c_index2:
                if dist(centroids[c_index], centroids[c_index2]) < chaining_thr:
                    chains[c_index].append(c_index2)

    old_chains = chains.copy()

    for iter in range(iterations):
        print(iter)
        used = np.zeros(len(chains))
        new_chains = []
        cnt = 0
        for i in range(len(chains)):

            if used[i] == 0:
                new_chains.append([])
                new_chains[cnt] = chains[i]
                used[i] = 1

                for e in chains[i]:
                    for j in range(len(chains)):
                        if j != i:
                            if used[j] == 0:
                                if e in chains[j]:
                                    new_chains[cnt].extend(chains[j])
                                    used[j] = 1
                                    break
                new_chains[cnt] = list(set(new_chains[cnt]))
                cnt += 1

        del chains
        chains = new_chains.copy()

    t = 1


    return chains


def chainer(data, centroids, labels, chaining_thr, max_iterations=10):
    ''' Chaining clusters

    Parameters
    ----------
    data : sequence of features
        input data to be clustered
    qbx_thresholds : sequence
        ist of distance thresholds for for QBX
    chaining_thr : float
        chaining threshold
    max_iterations : int

    '''

    clusters, clusters_density = clusters_from_labels(centroids, labels)

    all_centroids = centroids.copy()

    # array that has cluster numbers for each data point
    new_labels = np.zeros(len(data))
    # an array that has cluster numbers for each cluster centroid
    # Now we save centroid label information too
    new_clusters_ids = np.zeros(len(clusters))

    for i in range(len(clusters)):
        new_labels[clusters[i]] = i
        new_clusters_ids[i] = i  # same goes for cluster centroid numbers

    print("--------------------------------------------------------------")
    print("total number of initial clusters = ", len(list(set(new_labels))))

    f_start_time = time.time()

    # chaining clusters

    '''
    cluster_chaining is list of lists
    each list on ith index corresponds to the ith data point (now centroid)
    ith list will have all the cluster numbers ith data point (or ith centroid)
    belongs to

    '''
    iteration = max_iterations  # max number of iterations you want to run
    clusters_compact = list(set(new_labels))
    old_clusters = clusters_compact.copy()
    old_clusters_no = len(old_clusters)

    for ite in range(iteration):
        print('Iteration', ite)

        cluster_chaining = [[1]]
        for p in range(len(new_clusters_ids)):  # all_centroids
            # initially just assigning it the original
            # cluster numbers we got
            cluster_chaining.append([new_clusters_ids[p]])
        cluster_chaining.pop(0)

        for ck in range(len(clusters_compact)):  # cluster number

            cc = new_clusters_ids[ck]
            ct = all_centroids[ck]
            # should look at new centroids for this cluster too
            remaining_cts = np.argwhere(new_clusters_ids != cc)

            for k in remaining_cts:
                # k has point number that belongs to complement of cluster ck
                k = k[0]
                pointy = all_centroids[k]  # centroid value

                if dist(ct, pointy) <= chaining_thr:
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
                    remaining_points = np.argwhere(new_labels == j)
                    for k in remaining_points:
                        # k has point number that belongs to
                        # complement of cluster ck
                        k = k[0]
                        new_labels[k] = relabled[r]

                    remaining_points = np.argwhere(new_clusters_ids == j)
                    for k in remaining_points:
                        # k has centroid number that belongs to complement
                        # of cluster ck
                        k = k[0]
                        new_clusters_ids[k] = relabled[r]

                        # update cluster_chaining list of kth centroid too
                        cluster_chaining[k].append(relabled[r])

        new_clusters_no = len(list(set(new_labels)))

        print("total number of clusters = ", new_clusters_no)

        # overwritting a list with an integer
        if old_clusters_no == new_clusters_no:
            print("reached convergence")
            break
        old_clusters_no = new_clusters_no

    # print("--- %s seconds ---" % (time.time() - f_start_time))

    # print("total number of final clusters = ", len(list(set(new_labels))))
    # print("--------------------------------------------------------------")

    return new_labels
