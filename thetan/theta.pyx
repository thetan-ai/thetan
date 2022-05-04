# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp
# from scipy.linalg.cython_blas cimport ddot

from libc.math cimport fabs
cdef extern from "math.h" nogil:
    double fabs(double x)
    double sqrt(double x)

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t elsize)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

cdef extern from "limits.h":
    cdef int INT_MAX
    cdef unsigned int UINT_MAX

cdef extern from "float.h":
    cdef double DBL_MAX

ctypedef double (*metricp)(double *, double *, cnp.npy_intp) nogil

ctypedef struct Linkedlist:

    cnp.npy_intp value

    double dist

    Linkedlist * next


cdef double euclidean(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return sqrt(summ)


cdef double norm_ssd(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ / D


cdef double ssd(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ


cdef double norm_ssd_un(double * a, double * b, cnp.npy_intp D) nogil:
    cdef:
        int i
        int n = D // 4
        int rem = D % 4
        double result = 0

    # We manually unroll the loop for better cache optimization.
    for i in range(n):
        result += ((a[0] - b[0]) * (a[0] - b[0])
                  +(a[1] - b[1]) * (a[1] - b[1])
                  +(a[2] - b[2]) * (a[2] - b[2])
                  +(a[3] - b[3]) * (a[3] - b[3]))
        a += 4; b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result / D


def distance_matrix(double[:, ::1] X1, double[:, ::1] X2, metric='norm_ssd'):

    cdef:
        cnp.npy_intp i = 0
        cnp.npy_intp j = 0
        cnp.npy_intp I = X1.shape[0]
        cnp.npy_intp J = X2.shape[0]
        cnp.npy_intp D = X1.shape[1]
        metricp metric_func = &euclidean
        double[:, ::1] distance_matrix

    distance_matrix = np.zeros((I, J), dtype=np.float64)


    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    for i in range(I):
        for j in range(J):
            distance_matrix[i, j] = metric_func(&X1[i, 0], &X2[j, 0], D)

    return np.asarray(distance_matrix)


def thetan_neighb(double[:, ::1] X, double theta, metric='euclidean', all_out=False):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')

        cnp.npy_intp [:] counts = np.zeros(N, dtype='i8')
        cnp.npy_intp counts_prev
        Linkedlist ** near_clusters = <Linkedlist **> malloc(<size_t> N * sizeof(Linkedlist*))
        Linkedlist ** last_node = <Linkedlist **> malloc(<size_t> N * sizeof(Linkedlist*))
        Linkedlist * element = NULL
        Linkedlist * current = NULL
        Linkedlist * next = NULL


        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        cnp.npy_intp * nearc_value = NULL
        double * nearc_dist = NULL
        cnp.npy_intp cnt = 0
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
            centroids[d] = X[0, d]
        K = 1
        clusters_sizes[0] = 1.0
        counts[0] = 1
        near_clusters[0]
        near_clusters[0] = <Linkedlist *> malloc(sizeof(Linkedlist))
        near_clusters[0].value = 0
        near_clusters[0].dist = 0
        near_clusters[0].next = NULL
        last_node[0] = <Linkedlist *> near_clusters[0]

        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            counts_prev = counts[i]

            for k in range(K):

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < theta:

                    if dist < minimum:
                        minimum = dist
                        argmin = k

                    counts[i] += 1
                    if counts[i] == 1:
                        near_clusters[i] = <Linkedlist *> malloc(sizeof(Linkedlist))
                        near_clusters[i].value = k
                        near_clusters[i].dist = dist
                        near_clusters[i].next = NULL
                        last_node[i] = <Linkedlist *> near_clusters[i]
                    else:
                        element = <Linkedlist *> malloc(sizeof(Linkedlist))
                        element.value = k
                        element.dist = dist
                        element.next = NULL
                        last_node[i].next = element
                        last_node[i] = element

            # add in closest (argmin) cluster
            if counts[i] > counts_prev:
                indices[i] = argmin
                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]

            # create new cluster
            if counts[i] == counts_prev:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = X[i, d]
                clusters_sizes[K] = 1.0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                counts[i] += 1
                near_clusters[i] = <Linkedlist *> malloc(sizeof(Linkedlist))
                near_clusters[i].value = K
                near_clusters[i].dist = 0
                near_clusters[i].next = NULL
                last_node[i] = <Linkedlist *> near_clusters[i]
                K += 1


    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)
    final_counts = np.asarray(counts)
    final_counts_sum = final_counts.sum()

    nearc_dist = <double *> malloc(<size_t> final_counts_sum * sizeof(double))
    nearc_value = <cnp.npy_intp *> malloc(<size_t> final_counts_sum * sizeof(cnp.npy_intp))

    # Generate array from the linked list and simultaneously free memory

    cnt = 0
    for i in range(0, N):
        current = near_clusters[i]
        while current != NULL:
            nearc_dist[cnt] = current.dist
            nearc_value[cnt] = current.value
            next = current.next
            free(current)
            current = next
            cnt += 1
    free(near_clusters)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        final_nearc_value = np.asarray(<cnp.npy_intp[:final_counts_sum]> nearc_value).astype('i8')
        final_nearc_dist = np.asarray(<double[:final_counts_sum]> nearc_dist).astype('f8')
        return final_centroids, final_indices, final_summs, final_clusters_sizes, final_counts, final_nearc_value, final_nearc_dist
    else:
        free(summs)
        free(clusters_sizes)
        free(nearc_dist)
        free(nearc_value)
        return final_centroids, final_indices


def thetan_density(double[:, ::1] X, double theta, cnp.npy_intp B, double ratio, metric='euclidean', all_out=False):

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
        metricp metric_func = &euclidean
        double * clusters_densities_in = <double *> malloc(<size_t> 1 * sizeof(double))
        double * clusters_densities_out = <double *> malloc(<size_t> 1 * sizeof(double))
        double * clusters_ignore = <double *> malloc(<size_t> 1 * sizeof(double))


    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
            centroids[d] = X[0, d]
        K = 1
        clusters_sizes[0] = 1.0
        clusters_densities_in[0] = 1
        clusters_densities_out[0] = 0
        clusters_ignore[0] = 0

        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                # if clusters_ignore[k] == 1:
                #       continue

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin

                if dist < theta / 2.0:
                    clusters_densities_in[argmin] += 1
                else:
                    clusters_densities_out[argmin] += 1


            # create new cluster
            else:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))

                clusters_densities_in = <double *> realloc(clusters_densities_in, <size_t> (K + 1) * sizeof(double))
                clusters_densities_out = <double *> realloc(clusters_densities_out, <size_t> (K + 1) * sizeof(double))

                clusters_ignore = <double *> realloc(clusters_ignore, <size_t> (K + 1) * sizeof(double))

                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = X[i, d]
                clusters_sizes[K] = 1.0

                clusters_densities_in[K] = 1
                clusters_densities_out[K] = 0

                clusters_ignore[K] = 0

                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                K += 1

        for k in range(K):

            if clusters_densities_in[k] < ratio * clusters_densities_out[k]:

                clusters_ignore[k] = 1

    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        final_clusters_densities_in = np.asarray(<double[:K]> clusters_densities_in)
        final_clusters_densities_out = np.asarray(<double[:K]> clusters_densities_out)
        final_clusters_ignore = np.asarray(<double[:K]> clusters_ignore)
        return final_centroids, final_indices, final_summs, final_clusters_sizes, final_clusters_densities_in, final_clusters_densities_out, final_clusters_ignore
    else:
        free(summs)
        free(clusters_sizes)
        free(clusters_densities_in)
        free(clusters_densities_out)
        free(clusters_ignore)
        return final_centroids, final_indices


def theta_g1(double[:, ::1] X, double theta, metric='euclidean', all_out=False):

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
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
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

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin

            # create new cluster
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


def build_nice(centroids, labels):
    nice = np.ones(labels.shape[0], dtype=labels.dtype)
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    for c in clusters:
        if len(c) > 1:
            maxc = np.max(c)
            candidates.append(maxc)

    # deleting will change the labels of other Xs too which is not ideal
    # centroids_reduced = np.delete(final_centroids, candidates, axis=0)

    for i, f in enumerate(labels):
        for can in candidates:
            if labels[i] == can:
                nice[i] = 0

    return nice, candidates


def reassign_uncertain(double[:, ::1] X, double[:, ::1] centroids,
                       cnp.npy_intp[::1] labels, cnp.npy_intp[::1] nice,
                       metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        cnp.npy_intp k = 0, K = centroids.shape[0]
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        for i in range(N):

            minimum = DBL_MAX
            argmin = -1

            if nice[labels[i]] == 0:

                for k in range(K):

                    if nice[k] == 1:

                        dist = metric_func(&X[i, 0], &centroids[k, 0], D)

                        if dist < minimum:
                            minimum = dist
                            argmin = k

            indices[i] = argmin

    reassigned_indices = np.asarray(indices)

    return reassigned_indices


def reorder_labels(centroids, labels, nice, candidates):

    current_cidx = np.arange(centroids.shape[0])
    updated_cidx = current_cidx.copy()

    print(current_cidx)

    for c_i, c in enumerate(current_cidx):
        print(c_i, c)
        if c in candidates:
            updated_cidx[c_i + 1 :] -= 1

    print(updated_cidx)

    centroids_reduced = np.delete(centroids, candidates, axis=0)

    for c_i, c in enumerate(current_cidx):
        l_i = np.where(labels == c)[0]
        labels[l_i] = updated_cidx[c_i]

    return labels


def thetan_oic(double[:, ::1] X, double theta, metric='euclidean', all_out=False, ccentroids=True):

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
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
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

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin

            # create new cluster
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

    initial_centroids = np.asarray(<double[:K, :D]> centroids)
    initial_indices = np.asarray(indices)


    if ccentroids:
        # identify close centroids
        c_centroids, c_labels = thetan_oic(initial_centroids, theta=theta,
                                           metric=metric, all_out=False,
                                           ccentroids=False)

        nice, candidates = build_nice(c_centroids, c_labels)



    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        return initial_centroids, initial_indices, final_summs, final_clusters_sizes
    else:
        free(summs)
        free(clusters_sizes)
        return initial_centroids, initial_indices



def theta_g1w(double[:, ::1] X, double theta, metric='euclidean', double w=1.0, all_out=False):

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
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = w * X[0, d]
            centroids[d] = X[0, d]
        K = 1
        clusters_sizes[0] = w

        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin

            # create new cluster
            else:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = w * X[i, d]
                clusters_sizes[K] = w
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d] / w
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


def theta_delta(double[:, ::1] X, double theta, metric='euclidean', double w=1.0, double dw=1.0, all_out=False):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * deltas = <double *> malloc(<size_t> D * sizeof(double))
        double * deltas_xo = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = w * X[0, d]
            centroids[d] = X[0, d]
            deltas[d] = X[0, d]
            deltas_xo[d] = X[0, d]
        K = 1
        clusters_sizes[0] = w

        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                    deltas[argmin * D + d] += dw * (X[i, d] - deltas_xo[argmin * D + d])

                indices[i] = argmin

            # create new cluster
            else:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                deltas = <double *> realloc(deltas, <size_t> D * (K + 1) * sizeof(double))
                deltas_xo = <double *> realloc(deltas_xo, <size_t> D * (K + 1) * sizeof(double))

                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = w * X[i, d]
                clusters_sizes[K] = w
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d] / w
                    deltas[K * D + d] = X[i, d]
                    deltas_xo[K * D + d] = X[i, d]
                K += 1


    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_deltas = np.asarray(<double[:K, :D]> deltas)
    final_deltas_xo = np.asarray(<double[:K, :D]> deltas_xo)
    final_indices = np.asarray(indices)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        return final_centroids, final_indices, final_summs, final_clusters_sizes, final_deltas, final_deltas_xo
    else:
        free(summs)
        free(clusters_sizes)
        return final_centroids, final_indices, final_deltas



def assign(double[:, ::1] X, double[:, ::1] centroids, metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        cnp.npy_intp k = 0, K = centroids.shape[0]
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        for i in range(N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                dist = metric_func(&X[i, 0], &centroids[k, 0], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            indices[i] = argmin

    final_indices = np.asarray(indices)

    return final_indices


def gravity(double[:, ::1] X, double[:, ::1] centroids,
            cnp.npy_intp[:] sizes, double theta, metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        cnp.npy_intp k = 0, K = centroids.shape[0]
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')

        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

        cnp.npy_intp maximum = 0
        cnp.npy_intp argmax = -1

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        for i in range(N):
            maximum = 0
            argmax = -1
            for k in range(K):

                dist = metric_func(&X[i, 0], &centroids[k, 0], D)
                if dist < theta:
                    if sizes[k] > maximum:
                        maximum = sizes[k]
                        argmax = k

            indices[i] = argmax

    final_indices = np.asarray(indices)

    return final_indices


def calibrate(double[:, ::1] X, double[:, ::1] centroids, cnp.npy_intp[:] indices,
              cnp.npy_intp[:] flagged, metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        cnp.npy_intp k = 0, K = centroids.shape[0]
        # cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        for i in range(N):

            if flagged[i] == 1:

                minimum = DBL_MAX
                argmin = -1

                for k in range(K):

                    dist = metric_func(&X[i, 0], &centroids[k, 0], D)

                    if dist < minimum:
                        minimum = dist
                        argmin = k

                indices[i] = argmin

                # TODO: Update centroids

    final_indices = np.asarray(indices)

    return final_indices


def theta_g1t(double[:, ::1] X, double theta, metric='euclidean', all_out=False):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double * dtravel =  <double *> malloc(<size_t> 1 * sizeof(double))
        double * dtmp = <double *> malloc(<size_t> D * sizeof(double))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # first point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
            centroids[d] = X[0, d]
        K = 1
        clusters_sizes[0] = 1.0
        dtravel[0] = 0.0

        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                dist = metric_func(&X[i, 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    dtmp[d] = centroids[argmin * D + d]
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]

                dtravel[argmin] += metric_func(&dtmp[0], &centroids[argmin * D], D)


                indices[i] = argmin

            # create new cluster
            else:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                dtravel = <double *> realloc(dtravel, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = X[i, d]
                clusters_sizes[K] = 1.0
                dtravel[K] = 0.0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                K += 1


    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        free(dtmp)
        final_dtravel = np.asarray(<double[:K]> dtravel)
        return final_centroids, final_indices, final_summs, final_clusters_sizes, final_dtravel
    else:
        free(summs)
        free(clusters_sizes)
        free(dtmp)
        free(dtravel)
        return final_centroids, final_indices
