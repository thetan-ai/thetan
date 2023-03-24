# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False


cimport cython
import numpy as np
cimport numpy as cnp
# from scipy.linalg.cython_blas cimport ddot
from time import time

from libc.math cimport fabs
cdef extern from "math.h" nogil:
    double fabs(double x)
    double sqrt(double x)
    double floor(double x)

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
    cdef double DBL_MIN

ctypedef double (*metricp)(double *, double *, cnp.npy_intp) nogil

ctypedef struct Linkedlist:

    cnp.npy_intp value

    double dist

    Linkedlist * next


ctypedef struct Node:
    cnp.npy_intp sz
    cnp.npy_intp * elements



cdef double euclidean(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return sqrt(summ)


cdef double inf_norm(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double overall_max = DBL_MIN
        double tmp_max

    for d in range(D):
        tmp_max = fabs(ptr1[d] - ptr2[d])
        if tmp_max > overall_max:
            overall_max = tmp_max

    return overall_max


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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
        double cnt_orig = 0

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

                cnt_orig += 1
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


def theta_g1h(double[:, ::1] X, double theta, metric='euclidean', all_out=False):
    """ Adding trivial collision abilities
    """

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        cnp.npy_intp [:] multipliers = np.random.randint(100000, 200000, X.shape[1], dtype='i8')
        double sum_mult = <double>(np.sum(np.asarray(multipliers)))
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double * c_hash = <double *> malloc(<size_t> 1 * sizeof(double))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean
        double x_hash = 0
        double cnt_dist = 0, cnt_orig = 0
        double mdiv = 1 / (2 * theta)


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
        c_hash[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
            centroids[d] = X[0, d]
            c_hash[0] += floor(centroids[d] * mdiv) * multipliers[d]
        K = 1
        clusters_sizes[0] = 1.0


        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            x_hash = 0
            for d in range(D):
                x_hash += floor(X[i, d] * mdiv) * multipliers[d]

            for k in range(K):

                cnt_orig += 1

                if fabs(x_hash - c_hash[k]) <= sum_mult:

                    dist = metric_func(&X[i, 0], &centroids[k * D], D)
                    cnt_dist += 1

                    if dist < minimum:
                        minimum = dist
                        argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1.0
                c_hash[argmin] = 0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                    c_hash[argmin] += floor(centroids[argmin * D + d] * mdiv) * multipliers[d]
                indices[i] = argmin

            # create new cluster
            else:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                c_hash = <double *> realloc(c_hash, <size_t> (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = X[i, d]
                clusters_sizes[K] = 1.0
                c_hash[K] = 0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                    c_hash[K] += floor(centroids[K * D + d] * mdiv) * multipliers[d]
                K += 1


    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)

    print('Distance calls ', cnt_dist)
    print('Original calls ', cnt_orig)

    free(c_hash)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        return final_centroids, final_indices, final_summs, final_clusters_sizes
    else:
        free(summs)
        free(clusters_sizes)
        return final_centroids, final_indices

def theta_g1d(double[:, ::1] X, double theta, metric='euclidean', all_out=False):
    """ Adding trivial collision abilities
    """

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        # cnp.npy_intp [:] multipliers = np.random.randint(100000, 200000, X.shape[1], dtype='i8')
        # double sum_mult = np.float(np.sum(np.asarray(multipliers)))
        # cnp.npy_intp [:] X_hash = np.sum(np.intp(np.floor(np.asarray(X) / (theta * 2))) * np.asarray(multipliers), axis=1)
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double * x_bin = <double *> malloc(<size_t> D * sizeof(double))
        double * c_bin = <double *> malloc(<size_t> D * sizeof(double))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1, abort = 0
        metricp metric_func = &euclidean
        double cnt_dist = 0, cnt_orig = 0
        double mdiv = 1 / (2 * theta)


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
            c_bin[d] = floor(centroids[d] * mdiv)

        K = 1
        clusters_sizes[0] = 1.0


        for i in range(1, N):

            minimum = DBL_MAX
            argmin = -1

            for d in range(D):
                x_bin[d] = floor(X[i, d] * mdiv)

            for k in range(K):

                cnt_orig += 1

                for d in range(D):
                    if fabs(x_bin[d] - c_bin[k * D + d]) > 1:
                        abort = 1
                        break

                if abort == 1:
                    abort = 0
                    continue


                dist = metric_func(&X[i, 0], &centroids[k * D], D)
                cnt_dist += 1

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
                    c_bin[argmin * D + d] = floor(centroids[argmin * D + d] * mdiv)
                indices[i] = argmin

            # create new cluster
            else:

                summs = <double *> realloc(summs, <size_t> D * (K + 1) * sizeof(double))
                clusters_sizes = <double *> realloc(clusters_sizes, <size_t> (K + 1) * sizeof(double))
                centroids = <double *> realloc(centroids, <size_t> D * (K + 1) * sizeof(double))
                c_bin = <double *> realloc(c_bin, <size_t> D * (K + 1) * sizeof(double))
                indices[i] = K
                for d in range(D):
                    summs[K * D + d] = X[i, d]
                clusters_sizes[K] = 1.0
                for d in range(D):
                    centroids[K * D + d] = summs[K * D + d]
                    c_bin[K * D + d] = floor(centroids[K * D + d] * mdiv)
                K += 1


    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)

    print('Distance calls ', cnt_dist)
    print('Original calls ', cnt_orig)

    free(x_bin)
    free(c_bin)

    if all_out:
        final_summs = np.asarray(<double[:K, :D]> summs)
        final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
        return final_centroids, final_indices, final_summs, final_clusters_sizes
    else:
        free(summs)
        free(clusters_sizes)
        return final_centroids, final_indices



def candidate_labels_c(double[:, ::1] centroids, cnp.npy_intp[:] labels, cnp.npy_intp[:] initial_labels):
    """ Find labels for removal in cases of conflict and return
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


        # print if you need to print
        # for i in range(K):
        #    for j in range(clusters[i].sz):
        #        print(i, j, clusters[i].elements[j])


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
    # candidates = None

    return final_nice, final_candidates, clusters2, final_updated_labels


def candidate_labels_outliers_c(double[:, ::1] centroids, cnp.npy_intp[:] candidates, cnp.npy_intp[:] initial_labels):
    """ Find labels for removal in cases of outliers
    """

    cdef:

        cnp.npy_intp N = initial_labels.shape[0]
        cnp.npy_intp i = 0, j = 0, cnt = 0
        cnp.npy_intp N2 = candidates.shape[0]
        cnp.npy_intp [:] updated_labels = initial_labels.copy()
        cnp.npy_intp [:] nice = np.ones(initial_labels.shape[0], dtype=np.intp)

    with nogil:

        for i in range(N):
            for j in range(N2):
                if initial_labels[i] == candidates[j]:
                    nice[i] = 0
                    updated_labels[i] = -1

        for i in range(N):
            for j in range(N2):
                if initial_labels[i] > candidates[j]:
                    updated_labels[i] -= 1


    return np.asarray(nice), np.asarray(updated_labels)


def reassign_uncertain_update_labels(double[:, ::1] X, double[:, ::1] centroids,
                                     cnp.npy_intp[::1] labels, cnp.npy_intp[::1] nice,
                                     metric='euclidean'):
    # Reassign uncertain labels to closest cluster (note nice not currently used)

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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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


def theta_means(double[:, ::1] X, double[:, ::1] means_prev, double theta, metric='euclidean', all_out=False):

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
        metricp metric_func = &euclidean
        double * data = NULL


    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'inf_norm':
        metric_func = &inf_norm
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un

    with nogil:

        # Let's work first on integrating prior information
        # Here indices (labels) should not be updated

        data = &means_prev[0, 0]

        # first point becomes first cluster

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

            # add datum in existing cluster
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += data[i * D + d]
                clusters_sizes[argmin] += 1.0
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]

            # create new cluster
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

        # Now moving to process the actual data X
        # Here indices (labels) should be updated

        data = &X[0, 0]

        for i in range(0, N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                dist = metric_func(&data[i * D + 0], &centroids[k * D], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add datum in existing cluster
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += data[i * D + d]
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



def reassign_uncertain(double[:, ::1] X, double[:, ::1] centroids,
                       cnp.npy_intp[::1] labels, cnp.npy_intp[::1] nice,
                       metric='euclidean'):
    # Caution should be re-examined or deleted

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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    # Needs to be tested

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


def centroids_from_labels(double[:, ::1] X, cnp.npy_intp[:] indices):

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
    final_centroids = np.asarray(<double[:K, :D]> centroids)
    return np.ascontiguousarray(final_centroids)


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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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
    if metric == 'inf_norm':
        metric_func = &inf_norm
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


def img3d_to_ijkv_features(double[:, :, ::1] img3d):

    cdef:
        cnp.npy_intp i=0, j=0, k=0
        cnp.npy_intp I = img3d.shape[0], J = img3d.shape[1], K = img3d.shape[2]
        cnp.npy_intp cnt = 0
        double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))

    with nogil:

        for i in range(I):
            for j in range(J):
                for k in range(K):

                        features[cnt * 4 + 0] = i
                        features[cnt * 4 + 1] = j
                        features[cnt * 4 + 2] = k
                        features[cnt * 4 + 3] = img3d[i, j, k]
                        cnt += 1

    features_np = np.asarray(<double[:I * J * K, :4]> features)

    return features_np

