# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp

from robotikos.clustering import ClusterCentroid, ClusterMapCentroid
from robotikos.clustering import TreeCluster, TreeClusterMap


from libc.math cimport fabs
from cythonutils cimport Data2D, Shape, create_memview_2d, free_memview_2d

cdef extern from "math.h" nogil:
    double fabs(double x)

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t elsize)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

DTYPE = np.float32
DEF BIGGEST_DOUBLE = 1.7976931348623157e+308  # np.finfo('f8').max
DEF BIGGEST_INT = 2147483647  # np.iinfo('i4').max
DEF BIGGEST_FLOAT = 3.4028235e+38  # np.finfo('f4').max
DEF SMALLEST_FLOAT = -3.4028235e+38  # np.finfo('f4').max


cdef print_node(CentroidNode* node, prepend=""):
    if node == NULL:
        return ""
    cdef Data2D centroid
    centroid = <float[:node.centroid_shape.dims[0],:node.centroid_shape.dims[1]]> &node.centroid[0][0,0]
    txt = "{}".format(np.asarray(centroid).tolist())
    txt += " {" + ",".join(map(str, np.asarray(<int[:node.size]> node.indices))) + "}"
    txt += " children({})".format(node.nb_children)
    txt += " count({})".format(node.size)
    txt += " thres({})".format(node.threshold)
    txt += "\n"

    cdef int i
    for i in range(node.nb_children):
        txt += prepend
        if i == node.nb_children-1:
            # Last child
            txt += "`-- " + print_node(node.children[i], prepend + "    ")
        else:
            txt += "|-- " + print_node(node.children[i], prepend + "|   ")

    return txt


cdef void aabb_creation(Data2D streamline, float* aabb) nogil:
    """ Creates AABB enveloping the given streamline.

        Notes
        -----
        This currently assumes streamline is made of 3D points.
    """
    cdef:
        int N = streamline.shape[0], D = streamline.shape[1]
        int n, d
        float min_[3]
        float max_[3]

    for d in range(D):
        min_[d] = BIGGEST_FLOAT
        max_[d] = SMALLEST_FLOAT
        for n in range(N):

            if max_[d] < streamline[n, d]:
                max_[d] = streamline[n, d]

            if min_[d] > streamline[n, d]:
                min_[d] = streamline[n, d]

        aabb[d + 3] = (max_[d] - min_[d]) / 2.0 # radius
        aabb[d] = min_[d] + aabb[d + 3]  # center


cdef inline int aabb_overlap(float* aabb1, float* aabb2, float padding=0.) nogil:
    """ SIMD optimized AABB-AABB test

    Optimized by removing conditional branches
    """
    cdef:
        int x = fabs(aabb1[0] - aabb2[0]) <= (aabb1[3] + aabb2[3] + padding)
        int y = fabs(aabb1[1] - aabb2[1]) <= (aabb1[4] + aabb2[4] + padding)
        int z = fabs(aabb1[2] - aabb2[2]) <= (aabb1[5] + aabb2[5] + padding)

    return x & y & z


cdef CentroidNode* create_empty_node(Shape centroid_shape, float threshold) nogil:
    # Important: because the CentroidNode structure contains an uninitialized memview,
    # we need to zero-initialize the allocated memory (calloc or via memset),
    # otherwise during assignment CPython will try to call _PYX_XDEC_MEMVIEW on it and segfault.
    cdef CentroidNode* node = <CentroidNode*> calloc(1, sizeof(CentroidNode))
    node.centroid = create_memview_2d(centroid_shape.size, centroid_shape.dims)
        #node.updated_centroid = <float[:centroid_shape.dims[0], :centroid_shape.dims[1]]> calloc(centroid_shape.size, sizeof(float))

    node.father = NULL
    node.children = NULL
    node.nb_children = 0

    """
    TODO add polygonal box
    node.aabb[0] = 0
    node.aabb[1] = 0
    node.aabb[2] = 0
    node.aabb[3] = BIGGEST_FLOAT
    node.aabb[4] = BIGGEST_FLOAT
    node.aabb[5] = BIGGEST_FLOAT
    """
    node.threshold = threshold
    node.indices = NULL
    node.size = 0
    node.centroid_shape = centroid_shape
    return node




cdef class QuickBundlesX(object):

    def __init__(self, features_shape, levels_thresholds, Metric metric):
        
        # print('Start of QBX Init')
        self.metric = metric
        # print('features shape ', features_shape)
        # print(levels_thresholds)
        # self.features_shape = tuple2shape(features_shape)
        # TODO this below should eventually be renamed to features_size
        # only supporting Data2D
        cdef Shape fshape
        fshape.ndim = 2
        # print('fshape', features_shape)
        fshape.dims[0] = features_shape[0]
        fshape.dims[1] = features_shape[1]
        # print('dims 0', fshape.dims[0])
        # print('dims 1', fshape.dims[1])
        fshape.size = features_shape[0] * features_shape[1]
        self.features_shape = fshape
        self.nb_levels = len(levels_thresholds)
        # TODO add polygonal box


        # print('nv lev ', self.nb_levels)

        self.thresholds = <double*> calloc(self.nb_levels, sizeof(double))

        cdef int i
        for i in range(self.nb_levels):
            self.thresholds[i] = levels_thresholds[i]

        self.root = create_empty_node(self.features_shape, self.thresholds[0])

        self.level = None
        self.clusters = None
        self.stats.stats_per_layer = <QuickBundlesXStatsLayer*> calloc(self.nb_levels, sizeof(QuickBundlesXStatsLayer))
        # Important: because the CentroidNode structure contains an uninitialized memview,
        # we need to zero-initialize the allocated memory (calloc or via memset),
        # otherwise during assignment CPython will try to call _PYX_XDEC_MEMVIEW on it and segfault.
        # print('sinfos ', sizeof(StreamlineInfos))
        self.current_streamline = <StreamlineInfos*> calloc(1, sizeof(StreamlineInfos))

        # print('sz ', self.features_shape.size)
        # print('dims ',  self.features_shape.dims)

        self.current_streamline.features = create_memview_2d(self.features_shape.size, self.features_shape.dims)
        
        self.current_streamline.features_flip = create_memview_2d(self.features_shape.size,
                                                                  self.features_shape.dims)
        # print('End of QBX Init')

    def __dealloc__(self):
        # print('Starting dealloc')
        self.traverse_postorder(self.root, self._dealloc_node)
        self.root = NULL

        if self.thresholds != NULL:
            free(self.thresholds)
            self.thresholds = NULL

        if self.stats.stats_per_layer != NULL:
            free(self.stats.stats_per_layer)
            self.stats.stats_per_layer = NULL

        if self.current_streamline != NULL:
            free(self.current_streamline)
            self.current_streamline = NULL
        # print('End of dealloc')

    cdef int _add_child(self, CentroidNode* node) nogil:
        cdef double threshold = 0.0  # Leaf node doesn't need threshold.

        """
        with gil:
            # print('Starting add child')
            # print(node.level)
        """
        if node.level + 1 < self.nb_levels:
            threshold = self.thresholds[node.level + 1]

        cdef CentroidNode* child = create_empty_node(self.features_shape, threshold)
        child.level = node.level + 1

        # Add new child.
        child.father = node
        node.children = <CentroidNode**> realloc(node.children, (node.nb_children + 1) * sizeof(CentroidNode*))
        node.children[node.nb_children] = child
        node.nb_children += 1

        return node.nb_children - 1

    cdef void _update_node(self, CentroidNode* node, StreamlineInfos* streamline_infos) nogil:
        cdef Data2D element = streamline_infos.features[0]
        cdef int C = node.size
        cdef cnp.npy_intp n, d

        if streamline_infos.use_flip:
            element = streamline_infos.features_flip[0]

        # Update centroid
        cdef Data2D centroid = node.centroid[0]
        cdef cnp.npy_intp N = centroid.shape[0], D = centroid.shape[1]
        for n in range(N):
            for d in range(D):
                centroid[n, d] = ((centroid[n, d] * C) + element[n, d]) / (C+1)

        # Update list of indices
        node.indices = <int*> realloc(node.indices, (C+1)*sizeof(int))
        node.indices[C] = streamline_infos.idx
        node.size += 1

        # TODO add polygonal box
        # Update AABB
        # aabb_creation(centroid, node.aabb)

    cdef void _insert_in(self, CentroidNode* node, StreamlineInfos* streamline_infos, int[:] path) nogil:
        cdef:
            float dist, dist_flip
            cnp.npy_intp k
            NearestCluster nearest_cluster

        self._update_node(node, streamline_infos)

        if node.level == self.nb_levels:
            return

        nearest_cluster.id = -1
        nearest_cluster.dist = BIGGEST_DOUBLE
        nearest_cluster.flip = 0

        for k in range(node.nb_children):

            # TODO add polygonal box
            # Check streamline's aabb colides with the current child.
            # self.stats.stats_per_layer[node.level].nb_aabb_calls += 1
            # if aabb_overlap(node.children[k].aabb, streamline_infos.aabb, node.threshold):

            self.stats.stats_per_layer[node.level].nb_mdf_calls += 1
            dist = self.metric.c_dist(node.children[k].centroid[0], streamline_infos.features[0])

            # Keep track of the nearest cluster
            if dist < nearest_cluster.dist:
                nearest_cluster.dist = dist
                nearest_cluster.id = k
                nearest_cluster.flip = 0

            self.stats.stats_per_layer[node.level].nb_mdf_calls += 1
            dist_flip = self.metric.c_dist(node.children[k].centroid[0], streamline_infos.features_flip[0])

            if dist_flip < nearest_cluster.dist:
                nearest_cluster.dist = dist_flip
                nearest_cluster.id = k
                nearest_cluster.flip = 1

        if nearest_cluster.dist > node.threshold:
            # No near cluster, create a new one.
            nearest_cluster.id = self._add_child(node)

        streamline_infos.use_flip = nearest_cluster.flip
        path[node.level] = nearest_cluster.id
        self._insert_in(node.children[nearest_cluster.id], streamline_infos, path)

    cpdef object insert(self, Data2D datum, int datum_idx):

        # print('Starting insert')
        # # print('What the heck?', self.current_streamline.features) # [0]
        self.metric.feature.c_extract(datum, self.current_streamline.features[0])
        # print('First feature extract done')
        # print('Datum', datum)
        # CAREFULL the line bellow was added recently.
        self.metric.feature.c_extract(datum[::-1], self.current_streamline.features_flip[0])
        # print('Datum_flipped', datum[::-1])
        # print('Second feature extract done')
        self.current_streamline.idx = datum_idx
        # print('Middle of insert')
        # TODO add polygonal box
        # aabb_creation(self.current_streamline.features[0], self.current_streamline.aabb)
        path = -1 * np.ones(self.nb_levels, dtype=np.int32)
        self._insert_in(self.root, self.current_streamline, path)
        # print('Ending insert')
        return path

    def __str__(self):
        return print_node(self.root)

    cdef void traverse_postorder(self, CentroidNode* node, void (*visit)(QuickBundlesX, CentroidNode*)):
        cdef int i
        for i in range(node.nb_children):
            self.traverse_postorder(node.children[i], visit)
        visit(self, node)

    cdef void _dealloc_node(self, CentroidNode* node):
        free_memview_2d(node.centroid)

        if node.children != NULL:
            free(node.children)
            node.children = NULL

        free(node.indices)
        node.indices = NULL

        # No need to free node.father, only the current node.
        free(node)

    cdef object _build_tree_clustermap(self, CentroidNode* node):
        cdef Data2D centroid
        # centroid = <float[:self.features_shape.dims[0],:self.features_shape.dims[1]]> &node.centroid[0][0,0]
        centroid = <float[:self.features_shape.dims[0], :self.features_shape.dims[1]]> &node.centroid[0][0,0]

        tree_cluster = TreeCluster(threshold=node.threshold,
                                   centroid=np.asarray(centroid),
                                   indices=np.asarray(<int[:node.size]> node.indices).copy())
        cdef int i
        for i in range(node.nb_children):
            tree_cluster.add(self._build_tree_clustermap(node.children[i]))

        return tree_cluster

    def get_tree_cluster_map(self):
        return TreeClusterMap(self._build_tree_clustermap(self.root))

    def get_stats(self):
        stats_per_level = []
        for i in range(self.nb_levels):
            stats_per_level.append({'nb_mdf_calls': self.stats.stats_per_layer[i].nb_mdf_calls,
                                    # 'nb_aabb_calls': self.stats.stats_per_layer[i].nb_aabb_calls,
                                    'threshold': self.thresholds[i]})

        stats = {'stats_per_level': stats_per_level}

        return stats


cdef class Clusters:
    """ Provides Cython functionalities to interact with clustering outputs.

    This class allows to create clusters and assign elements to them.
    Assignements of a cluster are represented as a list of element indices.
    """
    def __init__(Clusters self):
        self._nb_clusters = 0
        self.clusters_indices = NULL
        self.clusters_size = NULL

    def __dealloc__(Clusters self):
        """ Deallocates memory created with `c_create_cluster` and `c_assign`. """
        for i in range(self._nb_clusters):
            free(self.clusters_indices[i])
            self.clusters_indices[i] = NULL

        free(self.clusters_indices)
        self.clusters_indices = NULL
        free(self.clusters_size)
        self.clusters_size = NULL

    cdef int c_size(Clusters self) nogil:
        """ Returns the number of clusters. """
        return self._nb_clusters

    cdef void c_assign(Clusters self, int id_cluster, int id_element, Data2D element) nogil except *:
        """ Assigns an element to a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster to which the element will be assigned.
        id_element : int
            Index of the element to assign.
        element : 2d array (float)
            Data of the element to assign.
        """
        cdef cnp.npy_intp C = self.clusters_size[id_cluster]
        self.clusters_indices[id_cluster] = <int*> realloc(self.clusters_indices[id_cluster], (C+1)*sizeof(int))
        self.clusters_indices[id_cluster][C] = id_element
        self.clusters_size[id_cluster] += 1

    cdef int c_create_cluster(Clusters self) nogil except -1:
        """ Creates a cluster and adds it at the end of the list.

        Returns
        -------
        id_cluster : int
            Index of the new cluster.
        """
        self.clusters_indices = <int**> realloc(self.clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self.clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self.clusters_size = <int*> realloc(self.clusters_size, (self._nb_clusters+1)*sizeof(int))
        self.clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1


cdef class ClustersCentroid(Clusters):
    """ Provides Cython functionalities to interact with clustering outputs
    having the notion of cluster's centroid.

    This class allows to create clusters, assign elements to them and
    update their centroid.

    Parameters
    ----------
    centroid_shape : int, tuple of int
        Information about the shape of the centroid.
    eps : float, optional
        Consider the centroid has not changed if the changes per dimension
        are less than this epsilon. (Default: 1e-6)
    """
    def __init__(ClustersCentroid self, centroid_shape, float eps=1e-6, *args, **kwargs):
        Clusters.__init__(self, *args, **kwargs)
        if isinstance(centroid_shape, int):
            centroid_shape = (1, centroid_shape)

        if not isinstance(centroid_shape, tuple):
            raise ValueError("'centroid_shape' must be a tuple or a int.")

        # TODO this is to be called centroid_shape
        # self._centroid_shape = tuple2shape(centroid_shape)
        self._centroid_shape = centroid_shape[-1]
        self.centroids = NULL
        self._updated_centroids = NULL
        self.eps = eps

    def __dealloc__(ClustersCentroid self):
        """ Deallocates memory created with `c_create_cluster` and `c_assign`.

        Notes
        -----
        The `__dealloc__` method of the superclass is automatically called:
        http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        """
        cdef cnp.npy_intp i
        for i in range(self._nb_clusters):
            free_memview_2d(self.centroids[i].features)
            free_memview_2d(self._updated_centroids[i].features)


        free(self.centroids)
        self.centroids = NULL
        free(self._updated_centroids)
        self._updated_centroids = NULL

    cdef void c_assign(ClustersCentroid self, int id_cluster, int id_element, Data2D element) nogil except *:
        """ Assigns an element to a cluster.

        In addition of keeping element's index, an updated version of the
        cluster's centroid is computed. The centroid is the average of all
        elements in a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster to which the element will be assigned.
        id_element : int
            Index of the element to assign.
        element : 2d array (float)
            Data of the element to assign.
        """
        cdef Data2D updated_centroid = self._updated_centroids[id_cluster].features[0]
        cdef cnp.npy_intp C = self.clusters_size[id_cluster]
        cdef cnp.npy_intp n, d

        cdef cnp.npy_intp N = updated_centroid.shape[0], D = updated_centroid.shape[1]
        for n in range(N):
            for d in range(D):
                updated_centroid[n, d] = ((updated_centroid[n, d] * C) + element[n, d]) / (C+1)

        Clusters.c_assign(self, id_cluster, id_element, element)

    cdef int c_update(ClustersCentroid self, cnp.npy_intp id_cluster) nogil except -1:
        """ Update the centroid of a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster of which its centroid will be updated.

        Returns
        -------
        int
            Tells whether the centroid has changed or not, i.e. converged.
        """
        cdef Data2D centroid = self.centroids[id_cluster].features[0]
        cdef Data2D updated_centroid = self._updated_centroids[id_cluster].features[0]
        cdef cnp.npy_intp N = updated_centroid.shape[0], D = centroid.shape[1]
        cdef cnp.npy_intp n, d
        cdef int converged = 1

        for n in range(N):
            for d in range(D):
                converged &= fabs(centroid[n, d] - updated_centroid[n, d]) < self.eps
                centroid[n, d] = updated_centroid[n, d]

        #cdef float * aabb = &self.centroids[id_cluster].aabb[0]

        # TODO add polygonal box
        # aabb_creation(centroid, self.centroids[id_cluster].aabb)

        return converged

    cdef int c_create_cluster(ClustersCentroid self) nogil except -1:
        """ Creates a cluster and adds it at the end of the list.

        Returns
        -------
        id_cluster : int
            Index of the new cluster.
        """
        self.centroids = <Centroid*> realloc(self.centroids, (self._nb_clusters+1) * sizeof(Centroid))
        # Zero-initialize the Centroid structure
        memset(&self.centroids[self._nb_clusters], 0, sizeof(Centroid))

        self._updated_centroids = <Centroid*> realloc(self._updated_centroids, (self._nb_clusters+1) * sizeof(Centroid))
        # Zero-initialize the new Centroid structure
        memset(&self._updated_centroids[self._nb_clusters], 0, sizeof(Centroid))

        self.centroids[self._nb_clusters].features = create_memview_2d(self._centroid_shape.size, self._centroid_shape.dims)
        self._updated_centroids[self._nb_clusters].features = create_memview_2d(self._centroid_shape.size, self._centroid_shape.dims)

        # Add polygonal box
        # aabb_creation(self.centroids[self._nb_clusters].features[0], self.centroids[self._nb_clusters].aabb)

        return Clusters.c_create_cluster(self)


def evaluate_aabb_checks():
    cdef:
        Data2D feature1 = np.array([[1, 0, 0], [1, 1, 0], [1 + np.sqrt(2)/2., 1 + np.sqrt(2)/2., 0]], dtype='f4')
        Data2D feature2 = np.array([[1, 0, 0], [1, 1, 0], [1 + np.sqrt(2)/2., 1 + np.sqrt(2)/2., 0]], dtype='f4') + np.array([0.5, 0, 0], dtype='f4')
        float[6] aabb1
        float[6] aabb2
        int res

    aabb_creation(feature1, &aabb1[0])
    aabb_creation(feature2, &aabb2[0])

    res = aabb_overlap(&aabb1[0], &aabb2[0])

    return np.asarray(aabb1), np.asarray(aabb2), res

