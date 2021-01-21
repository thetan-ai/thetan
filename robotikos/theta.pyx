# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp

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


cdef double euclidean(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return sqrt(summ)


cdef double ssd(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ


def theta_g1(double[:, ::1] X, double theta, metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        cnp.npy_intp * clusters_sizes = <cnp.npy_intp *> malloc(<size_t> 1 * sizeof(cnp.npy_intp))
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean

    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd

    with nogil:

        # firt point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
            centroids[d] = X[0, d]
        K = 1
        clusters_sizes[0] = 1

        for i in range(1, N):

            minimum = DBL_MAX
            argmin = 0

            for k in range(K):
                
                dist = ssd(&X[i, 0], &centroids[k * D], D)
         
                if dist < minimum:
                    minimum = dist
                    argmin = k

            # add in existing
            if minimum < theta:

                for d in range(D):
                    summs[argmin * D + d] += X[i, d]
                clusters_sizes[argmin] += 1
                for d in range(D):
                    centroids[argmin * D + d] = summs[argmin * D + d] / clusters_sizes[argmin]
                indices[i] = argmin

            # create new cluster
            else:

                K += 1
                summs = <double *> realloc(summs, <size_t> D * K * sizeof(double))
                clusters_sizes = <cnp.npy_intp *> realloc(clusters_sizes, <size_t> K * sizeof(cnp.npy_intp))
                centroids = <double *> realloc(centroids, <size_t> D * K * sizeof(double))
                indices[i] = K - 1
                for d in range(D):
                    summs[(K - 1) * D + d] = X[i, d]
                clusters_sizes[K - 1] = 1
                for d in range(D):
                    centroids[(K - 1) * D + d] = summs[(K - 1) * D + d] 

    final_summs = np.asarray(<double[:K, :D]> summs)
    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_clusters_sizes = np.asarray(<cnp.npy_intp[:K]> clusters_sizes)
    final_indices = np.asarray(indices)
    
    return final_summs, final_clusters_sizes, final_centroids, final_indices

