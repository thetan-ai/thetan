# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport ddot

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


def theta_g1(double[:, ::1] X, double theta, metric='euclidean'):

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

    final_summs = np.asarray(<double[:K, :D]> summs)
    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_clusters_sizes = np.asarray(<double[:K]> clusters_sizes).astype('i8')
    final_indices = np.asarray(indices)
    
    return final_summs, final_clusters_sizes, final_centroids, final_indices

