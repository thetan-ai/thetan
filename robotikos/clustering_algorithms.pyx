# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

# from dipy.utils.six.moves import xrange
import itertools
import numpy as np
cimport numpy as cnp
from cythonutils cimport Data2D, Shape
from metricspeed cimport Metric
from clusteringspeed cimport ClustersCentroid, Centroid, QuickBundlesX #, Shape
from robotikos.clustering import ClusterMapCentroid, ClusterCentroid

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)


DTYPE = np.float32
DEF BIGGEST_DOUBLE = 1.7976931348623157e+308  # np.finfo('f8').max
DEF BIGGEST_FLOAT = 3.4028235e+38  # np.finfo('f4').max
DEF BIGGEST_INT = 2147483647  # np.iinfo('i4').max


def clusters_centroid2clustermap_centroid(ClustersCentroid clusters_list):
    """ Converts a `ClustersCentroid` object (Cython) to a `ClusterMapCentroid`
    object (Python).

    Only basic functionalities are provided with a `Clusters` object. To have
    more flexibility, one should use `ClusterMap` object, hence this conversion
    function.

    Parameters
    ----------
    clusters_list : `ClustersCentroid` object
        Result of the clustering contained in a Cython's object.

    Returns
    -------
    `ClusterMapCentroid` object
        Result of the clustering contained in a Python's object.
    """
    clusters = ClusterMapCentroid()
    cdef Data2D features
    shape = clusters_list._centroid_shape

    for i in xrange(clusters_list._nb_clusters):
        features = <float[:shape.dims[0], :shape.dims[1]]> \
            &clusters_list.centroids[i].features[0][0,0]
        centroid = np.asarray(features)
        indices = np.asarray(<int[:clusters_list.clusters_size[i]]> clusters_list.clusters_indices[i]).tolist()
        clusters.add_cluster(ClusterCentroid(id=i, centroid=centroid, indices=indices))

    return clusters


def peek(iterable):
    """ Returns the first element of an iterable and the iterator. """
    iterable = iter(iterable)
    first = next(iterable, None)
    iterator = itertools.chain([first], iterable)
    return first, iterator


def quickbundlesx(streamlines, Metric metric, thresholds, ordering=None):
    """ Clusters streamlines using QuickBundlesX.

    Parameters
    ----------
    streamlines : list of 2D arrays
        List of streamlines to cluster.
    metric : `Metric` object
        Tells how to compute the distance between two streamlines.
    thresholds : list of double
        Thresholds to use for each clustering layer. A threshold represents the
        maximum distance from a cluster for a streamline to be still considered
        as part of it.
    ordering : iterable of indices, optional
        Iterate through `data` using the given ordering.

    Returns
    -------
    `TreeClusterMap` object
        Result of the clustering. Use get_clusters() to get the clusters at
        a specific level of the hierarchy.

    References
    ----------

    .. [Garyfallidis16] Garyfallidis E. et al. QuickBundlesX: Sequential
                    clustering of millions of streamlines in multiple
                    levels of detail at record execution time. Proceedings
                    of the, International Society of Magnetic Resonance
                    in Medicine (ISMRM). Singapore, 4187, 2016.

     .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    if ordering is None:
        ordering = xrange(len(streamlines))

    # Check if `ordering` or `streamlines` are empty
    first_idx, ordering = peek(ordering)
    if first_idx is None or len(streamlines) == 0:
        return ClusterMapCentroid()

    features_shape = streamlines[first_idx].shape
    # print('First idx shape', features_shape)
    cdef QuickBundlesX qbx = QuickBundlesX(features_shape, thresholds, metric)
    cdef int idx

    # print('Inside >>>')
    for idx in ordering:
        # print('Start of the loop')
        # print(idx)
        streamline = streamlines[idx]
        if not streamline.flags.writeable or streamline.dtype != DTYPE:
            streamline = streamline.astype(DTYPE)
        # print('insert')
        qbx.insert(streamline, idx)
        # print('End of the loop')

    return qbx.get_tree_cluster_map()


def fast_grouping_node(streamlines, Metric metric, thresholds, ordering=None):
    """ Clusters streamlines using FGN.

    Parameters
    ----------
    streamlines : list of 2D arrays
        List of streamlines to cluster.
    metric : `Metric` object
        Tells how to compute the distance between two streamlines.
    thresholds : list of double
        Thresholds to use for each clustering layer. A threshold represents the
        maximum distance from a cluster for a streamline to be still considered
        as part of it.
    ordering : iterable of indices, optional
        Iterate through `data` using the given ordering.

    Returns
    -------
    `TreeClusterMap` object
        Result of the clustering. Use get_clusters() to get the clusters at
        a specific level of the hierarchy.

    """
    if ordering is None:
        ordering = xrange(len(streamlines))

    # Check if `ordering` or `streamlines` are empty
    first_idx, ordering = peek(ordering)
    if first_idx is None or len(streamlines) == 0:
        return ClusterMapCentroid()

    features_shape = streamlines[first_idx].shape
    # print('First idx shape', features_shape)
    cdef QuickBundlesX qbx = QuickBundlesX(features_shape, thresholds, metric)
    cdef int idx

    # print('Inside >>>')
    for idx in ordering:
        # print('Start of the loop')
        # print(idx)
        streamline = streamlines[idx]
        if not streamline.flags.writeable or streamline.dtype != DTYPE:
            streamline = streamline.astype(DTYPE)
        # print('insert')
        qbx.insert(streamline, idx)
        # print('End of the loop')

    return qbx.get_tree_cluster_map()