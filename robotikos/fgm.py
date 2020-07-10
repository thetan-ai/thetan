import numpy as np
from robotikos.clustering import FastGroupingNode, ClusterMapCentroid, ClusterCentroid
from time import time


def fgn_merging_level(fgn_online_all_levels, level,
                      thresholds, metric, data, rng):

    thr = thresholds[level - 1]

    fgn_merge = FastGroupingNode([thr], metric=metric)

    # final_level = len(thresholds)
    fgn_clusters = fgn_online_all_levels.get_clusters(level)
    len_fgn_fl = len(fgn_clusters)

    # this random selection will be only effective to the centroids of
    # the selected level
    fgn_ordering_final = rng.choice(len_fgn_fl, len_fgn_fl, replace=False)

    fgn_merged_cluster_map = fgn_merge.cluster(
        fgn_clusters.centroids,
        ordering=fgn_ordering_final).get_clusters(1)

    fgn_cluster_map = fgn_clusters

    merged_cluster_map = ClusterMapCentroid()

    for cluster in fgn_merged_cluster_map:
        merged_cluster = ClusterCentroid(centroid=cluster.centroid)
        for i in cluster.indices:
            merged_cluster.indices.extend(fgn_cluster_map[i].indices)
        merged_cluster_map.add_cluster(merged_cluster)

    merged_cluster_map.refdata = data

    return merged_cluster_map


def fgn(data, thresholds, metric,
        select_randomly=None, rng=None, verbose=True):
    """ Fast Grouping Node

    Parameters
    ----------
    data : sequence of features
    thresholds : sequence
        List of distance thresholds for FGN
    metric : class
    nb_pts : int
        Number of points for discretizing each streamline
    select_randomly : int
        Randomly select a specific number of streamlines. If None all the
        streamlines are used.
    rng : RandomState
        If None then RandomState is initialized internally.
    verbose : bool, optional.
        If True, log information. Default False.

    Returns
    -------
    all_levels : contains clusters for each level
        Contains the clusters of all levels.
    """

    if verbose:
        print('In FGN')
        print('Thresholds ', thresholds)

    len_s = len(data)
    if select_randomly is None:
        select_randomly = len_s

    if rng is None:
        rng = np.random.RandomState()

    # TODO implement random selection
    # indices = rng.choice(len_s, min(select_randomly, len_s),
    #                      replace=False)

    if verbose:
        print(' FGN phase starting...')

    fgn = FastGroupingNode(thresholds, metric)

    feature_shape = data[0].shape

    fgn_next = fgn.cluster_online(feature_shape)

    for i in range(len(data)):
        fgn_online_tree, _path = fgn_next(data[i], i)

    fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

    # return fgn_online_all_levels
    all_levels = {}
    for l in range(len(thresholds)):
        all_levels[l + 1] = fgn_online_all_levels.get_clusters(l + 1)

    return all_levels


# @profile
def fgn_ma(data, thresholds, metric,
           select_randomly=None, rng=None, verbose=True):
    """ Fast Grouping Node that Merges (re-clusters) all Levels of the tree

    Parameters
    ----------
    data : sequence of features
    thresholds : sequence
        List of distance thresholds for FGN
    metric : class
    nb_pts : int
        Number of points for discretizing each streamline
    select_randomly : int
        Randomly select a specific number of streamlines. If None all the
        streamlines are used.
    rng : RandomState
        If None then RandomState is initialized internally.
    verbose : bool, optional.
        If True, log information. Default False.

    Returns
    -------
    all_levels : contains clusters for each level
        Contains the clusters of all levels.
    """

    if verbose:
        print('In FGN MA')
        print('Thresholds ', thresholds)

    t = time()
    len_s = len(data)
    if select_randomly is None:
        select_randomly = len_s

    if rng is None:
        rng = np.random.RandomState()

    # TODO implement random selection
    # indices = rng.choice(len_s, min(select_randomly, len_s),
    #                      replace=False)

    if verbose:
        print(' FGN phase starting...')

    fgn = FastGroupingNode(thresholds, metric)

    feature_shape = data[0].shape

    fgn_next = fgn.cluster_online(feature_shape)

    for i in range(len(data)):
        fgn_online_tree, _path = fgn_next(data[i], i)

    fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

    if verbose:
        print(' Merging phase starting ...')

    all_levels = {}
    for l in range(len(thresholds)):
        if verbose:
            print(' Before merging layer %d has %d' %
                  (l + 1, len(fgn_online_all_levels.get_clusters(l + 1))))

        merged_cluster_map = fgn_merging_level(fgn_online_all_levels,
                                               l + 1, thresholds,
                                               metric, data,
                                               rng=rng)

        all_levels[l + 1] = merged_cluster_map

        if verbose:
            print(' After merging layer %d has %d' %
                  (l + 1, len(merged_cluster_map)))

    if verbose:
        print(' Duration %0.3f sec. \n' % (time() - t,))

    return all_levels


def assign_to_centroids(data, centroids, metric):
    """ Assign data to nearest centroids """
    y_pred = []
    for x in data:
        min_dists = []
        for i in range(len(centroids)):
            c = centroids[i]
            d = metric.dist(x.astype('f4'), c.astype('f4'))
            min_dists.append(d)
        y_pred.append(np.argmin(min_dists))
    return y_pred


def labels_to_centroids(data, labels):

    labels = np.array(labels)
    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)
    centroids = {}
    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids

# @profile
def fgn_shuffles(data, thresholds, metric,
                 num_shuffles,
                 mode='fgn',
                 verbose=False):

    class DataSequence2D:
        def __init__(self, data, indices):

            self.indices = indices
            self.data = data

        def __getitem__(self, i):

            return self.data[self.indices[i]]

        def __len__(self):
            return len(self.indices)

    internal_indices = np.arange(len(data))

    # np.random.seed(42)

    collect = {}

    for i in range(num_shuffles):

        if verbose:
            print('Shuffle ', i, ' Mode ', )

        np.random.shuffle(internal_indices)

        ds = DataSequence2D(data, internal_indices)

        if verbose:
            print('Calling ', mode)

        if mode == 'fgn':
            all_levels = fgn(ds, thresholds, metric=metric, verbose=verbose)
        elif mode == 'fgn_ma':
            all_levels = fgn_ma(ds, thresholds, metric=metric, verbose=verbose)
        else:
            raise ValueError('Incorrect mode')

        collect[i] = {}

        if verbose:
            print('Collecting all centroids of all levels')

        for j in range(len(thresholds)):

            collect[i][j + 1] = all_levels[j + 1].centroids

    all_levels_centroids = {}

    final_levels = {}

    if verbose:
        print('Reclustering centroids and reassigning all data to each level')

    for j in range(len(thresholds)):
        all_levels_centroids[j + 1] = []
        for i in range(num_shuffles):
            all_levels_centroids[j + 1].extend(collect[i][j + 1])
        res = fgn(all_levels_centroids[j + 1], [thresholds[j]], metric,
                  verbose=verbose)
        new_centroids = res[1].centroids
        labels = assign_to_centroids(data, new_centroids, metric=metric)
        final_levels[j + 1] = {}
        final_levels[j + 1]['labels'] = labels
        final_levels[j + 1]['centroids'] = labels_to_centroids(data, labels)

    return final_levels

