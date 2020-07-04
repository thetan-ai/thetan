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

    print('In FGN')
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

    # return fgn_online_all_levels
    all_levels = {}
    for l in range(len(thresholds)):
        all_levels[l + 1] = fgn_online_all_levels.get_clusters(l + 1)

    return all_levels


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

