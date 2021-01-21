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


cdef double euclidean(double * ptr1, double * ptr2, cnp.npy_intp D) nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return sqrt(summ)


def theta_g1(double[:, ::1] X, double theta):
    
    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D)
        cnp.npy_intp * clusters_sizes = <cnp.npy_intp *> malloc(<size_t> 1)  
        double * centroid = <double *> malloc(<size_t> D)

    with nogil:

        # firt point becomes first cluster
        indices[0] = 0
        for d in range(D):
            summs[d] = X[0, d]
        K = 1
        clusters_sizes[0] = 1
        
        for i in range(1, N):

            for k in range(K):

                for d in range(D):
                    centroid[d] = summs[k * D + d] / clusters_sizes[k]

                if euclidean(&X[i, 0], &centroid[0], D) < theta:

                    for d in range(D):
                    
                        summs[k * D + d] += X[i, d]

                    clusters_sizes[k] += 1

                
            
    final_summs = np.asarray(<double[:K, :D]>summs)
    final_clusters_sizes = np.asarray(<cnp.npy_intp[:K]>clusters_sizes)

    free(centroid)

    return final_summs, final_clusters_sizes    

