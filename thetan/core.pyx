# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp

from thetan.metrics cimport metricp,get_metric_func

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t elsize)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)

cdef extern from "limits.h":
    cdef int INT_MAX

cdef extern from "float.h":
    cdef double DBL_MAX


ctypedef struct Node:
    cnp.npy_intp sz
    cnp.npy_intp * elements


def sequential(double[:, ::1] X, double theta, metric='euclidean', all_out=False, verbose=False):
    """Cluster data in one pass with Thetan Sequential (TS).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data to cluster, as a C-contiguous float64 array with at least
        one row.
    theta : float
        Distance threshold. A point joins its nearest centroid only when
        the distance is strictly less than theta.
    metric : str, optional
        Distance metric, any name accepted by distance_matrix. Default is
        'euclidean'. 'dot_prod' is the negative dot product, so theta is
        normally negative: theta=-0.9 means a point joins a centroid when
        their dot product is above 0.9. The dot product also grows with
        the length of the vectors, so rows are usually normalised to unit
        length first.
    all_out : bool, optional
        If True, also return the sum and size of each cluster. Default is
        False.
    verbose : bool, optional
        If True, print the number of distance evaluations. Default is
        False.

    Returns
    -------
    final_centroids : ndarray, shape (n_clusters, n_features)
        Mean of the points in each cluster.
    final_indices : ndarray, shape (n_samples,)
        Cluster index of each point.
    final_summs : ndarray, shape (n_clusters, n_features)
        Sum of the points in each cluster. Only returned if all_out is
        True.
    final_clusters_sizes : ndarray, shape (n_clusters,)
        Number of points in each cluster. Only returned if all_out is
        True.
    """

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = get_metric_func(metric)
        double cnt_orig = 0

    with nogil:
        indices[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
            centroids[d] = X[0, d]
        K = 1
        clusters_sizes[0] = 1.0

        for i in range(1, N):
            minimum = DBL_MAX
            argmin = -1

            for k in range(K):
                cnt_orig += 1
                dist = metric_func(&X[i, 0], &centroids[k * D], D)
                if dist < minimum:
                    minimum = dist
                    argmin = k

            if minimum < theta:
                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin

            else:
                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = X[i, d]
                clusters_sizes[K] = 1.0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                K += 1


    if verbose:
        print('Original calls ', cnt_orig)

    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        return final_centroids, final_indices, final_summs, final_clusters_sizes
    else:
        free(summs)
        free(clusters_sizes)
        return final_centroids, final_indices


def candidate_labels_c(double[:, ::1] centroids, cnp.npy_intp[:] labels, cnp.npy_intp[:] initial_labels):
    """Find duplicate centroids to delete after a TS pass.

    Running TS on the centroids of a TS pass puts centroids that lie
    within theta of each other in the same group. In each group the
    centroid with the smallest index is kept and every other one becomes a
    candidate for deletion. Point labels are then shifted so they index
    the centroid array after the candidates are removed with np.delete.

    Parameters
    ----------
    centroids : ndarray, shape (n_groups, n_features)
        Centroids from TS on the centroids. Not used.
    labels : ndarray, shape (n_centroids,)
        Group index of each first-pass centroid, as returned by TS on the
        centroids.
    initial_labels : ndarray, shape (n_samples,)
        First-pass label of each point.

    Returns
    -------
    final_nice : ndarray, shape (n_samples,)
        1 if the point's centroid is kept, 0 if it is deleted.
    final_candidates : ndarray, shape (n_candidates,)
        Indices of the first-pass centroids to delete. Not sorted.
    clusters2 : None
        Always None. A placeholder so callers can unpack four values.
    final_updated_labels : ndarray, shape (n_samples,)
        Label of each point after deletion. Points whose centroid is
        deleted get a negative value (not always -1) and need to be
        reassigned with reassign_uncertain_update_labels.
    """

    cdef:
        cnp.npy_intp N = initial_labels.shape[0]
        cnp.npy_intp i = 0, j = 0, cnt = 0
        cnp.npy_intp K = labels.shape[0]
        cnp.npy_intp [:] updated_labels = initial_labels.copy()
        cnp.npy_intp [:] nice = np.ones(initial_labels.shape[0], dtype=np.intp)
        Node * clusters = <Node *> malloc(<size_t> K * sizeof(Node))
        cnp.npy_intp * candidates = NULL
        cnp.npy_intp minc = INT_MAX


    with nogil:
        for i in range(K):
            clusters[i].sz = 0
            clusters[i].elements = NULL

        for i in range(K):
            clusters[labels[i]].sz += 1
            clusters[labels[i]].elements = <cnp.npy_intp *> realloc(clusters[labels[i]].elements, <size_t> clusters[labels[i]].sz * sizeof(cnp.npy_intp))
            clusters[labels[i]].elements[clusters[labels[i]].sz - 1] = i

        cnt = 0
        for i in range(K):
            minc = INT_MAX

            for j in range(clusters[i].sz):
                if clusters[i].elements[j] < minc:
                    minc = clusters[i].elements[j]

            for j in range(clusters[i].sz):
                if clusters[i].elements[j] > minc:
                    candidates = <cnp.npy_intp *> realloc(candidates, <size_t> (cnt + 1) * sizeof(cnp.npy_intp))
                    candidates[cnt] = clusters[i].elements[j]
                    cnt += 1

        for i in range(K):
            free(clusters[i].elements)
        free(clusters)

        if cnt > 0:
            for i in range(N):
                for j in range(cnt):
                    if initial_labels[i] == candidates[j]:
                        nice[i] = 0
                        updated_labels[i] = -1

            for i in range(N):
                for j in range(cnt):
                    if initial_labels[i] > candidates[j]:
                        updated_labels[i] -= 1

    if cnt > 0:
        final_candidates = np.asarray(<cnp.npy_intp[:cnt]> candidates)
    else:
        final_candidates = np.array([], dtype=np.intp)

    final_nice = np.asarray(nice)
    final_updated_labels = np.asarray(updated_labels)
    clusters2 = None

    return final_nice, final_candidates, clusters2, final_updated_labels


def reassign_uncertain_update_labels(double[:, ::1] X, double[:, ::1] centroids,
                                     cnp.npy_intp[::1] labels, cnp.npy_intp[::1] nice,
                                     metric='euclidean'):
    """Give every point with a negative label its closest centroid.

    Points with a non-negative label keep it. Points with a negative label
    (their centroid was deleted) get the index of the nearest centroid,
    with no theta check.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data, as a C-contiguous float64 array.
    centroids : ndarray, shape (n_clusters, n_features)
        Centroids left after deletion.
    labels : ndarray, shape (n_samples,)
        Labels from candidate_labels_c. Negative values mark the points to
        reassign.
    nice : ndarray, shape (n_samples,)
        Not used.
    metric : str, optional
        Distance metric, any name accepted by distance_matrix. Default is
        'euclidean'. 'dot_prod' is the negative dot product, so the
        closest centroid is the one with the largest dot product.

    Returns
    -------
    reassigned_indices : ndarray, shape (n_samples,)
        New labels with no negative values.
    """
    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        cnp.npy_intp k = 0, K = centroids.shape[0]
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = get_metric_func(metric)

    with nogil:
        for i in range(N):
            minimum = DBL_MAX
            argmin = -1
            if labels[i] < 0:
                for k in range(K):
                    dist = metric_func(&X[i, 0], &centroids[k, 0], D)
                    if dist < minimum:
                        minimum = dist
                        argmin = k
                indices[i] = argmin
            else:
                indices[i] = labels[i]

    reassigned_indices = np.asarray(indices)

    return reassigned_indices


def sequential_prelim(double[:, ::1] X, double[:, ::1] means_prev, double theta, metric='euclidean', all_out=False):
    """Run TS on data with prior centroids placed in front of it.

    The prior centroids in means_prev are processed first with TS, so any
    that lie within theta of each other merge, and each one counts as a
    single sample in the running means. The points in X are then
    processed in order against those clusters, exactly as in sequential.
    This is the second pass of Thetan Berserker.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data to cluster, as a C-contiguous float64 array.
    means_prev : ndarray, shape (n_prev, n_features)
        Prior centroids, with at least one row and the same number of
        columns as X.
    theta : float
        Distance threshold. A point joins its nearest centroid only when
        the distance is strictly less than theta.
    metric : str, optional
        Distance metric, any name accepted by distance_matrix. Default is
        'euclidean'. 'dot_prod' is the negative dot product, so theta is
        normally negative (see sequential).
    all_out : bool, optional
        If True, also return the sum and size of each cluster. Default is
        False.

    Returns
    -------
    final_centroids : ndarray, shape (n_clusters, n_features)
        Mean of each cluster, including the prior centroids it absorbed.
    final_indices : ndarray, shape (n_samples,)
        Cluster index of each point in X. The prior centroids get no
        labels.
    final_summs : ndarray, shape (n_clusters, n_features)
        Sum of each cluster's members, prior centroids included. Only
        returned if all_out is True.
    final_clusters_sizes : ndarray, shape (n_clusters,)
        Number of members in each cluster, with each prior centroid
        counted once. Only returned if all_out is True.
    """

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp NE = means_prev.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = get_metric_func(metric)
        double * data = NULL


    with nogil:
        data = &means_prev[0, 0]
        for d in range(D):
            summs[d] = data[0 * D + d]
            centroids[d] = data[0 * D + d]
        K = 1
        clusters_sizes[0] = 1.0

        for i in range(1, NE):
            minimum = DBL_MAX
            argmin = -1
            for k in range(K):
                dist = metric_func(&data[i * D + 0], &centroids[k * D], D)
                if dist < minimum:
                    minimum = dist
                    argmin = k
            if minimum < theta:
                for d in range(D):
                    summs[argmin * D + d] += data[i * D + d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
            else:
                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                for d in range(D):
                    summs[K * D + d] = data[i * D + d]
                clusters_sizes[K] = 1.0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                K += 1

        data = &X[0, 0]
        for i in range(0, N):
            minimum = DBL_MAX
            argmin = -1
            for k in range(K):
                dist = metric_func(&data[i * D + 0], &centroids[k * D], D)
                if dist < minimum:
                    minimum = dist
                    argmin = k
            if minimum < theta:
                for d in range(D):
                    summs[argmin * D + d] += data[i * D + d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin
            else:
                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = data[i * D + d]
                clusters_sizes[K] = 1.0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                K += 1

    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)
    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        return final_centroids, final_indices, final_summs, final_clusters_sizes
    else:
        free(summs)
        free(clusters_sizes)
        return final_centroids, final_indices


def centroids_from_labels(const double[:, ::1] X, cnp.npy_intp[:] indices):
    """Compute the mean of the points in each cluster.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data, as a C-contiguous float64 array.
    indices : ndarray, shape (n_samples,)
        Cluster label of each point. Every value must be non-negative;
        this is not checked.

    Returns
    -------
    np.ascontiguousarray(final_centroids) : ndarray, shape (max(indices) + 1, n_features)
        Mean of the points with each label. A label in that range with no
        points gets a row of NaN.
    """

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp K = np.max(indices) + 1
        cnp.npy_intp i = 0, d = 0, k = 0
        double * centroids = <double *> calloc(<size_t> K, <size_t> D * sizeof(double))
        double * cluster_sizes = <double *> calloc(<size_t> K, <size_t> sizeof(double))

    with nogil:
        for i in range(N):
            for d in range(D):
                centroids[indices[i] * D + d] += X[i, d]
            cluster_sizes[indices[i]] += 1.0
        for k in range(K):
            for d in range(D):
                centroids[k * D + d] /= cluster_sizes[k]

    free(cluster_sizes)
    final_centroids = np.asarray(<double[:K, :D]> centroids, dtype=np.float64)
    return np.ascontiguousarray(final_centroids)