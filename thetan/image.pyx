# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp
# from scipy.linalg.cython_blas cimport ddot


from libc.math cimport fabs
cdef extern from "math.h" nogil:
    double fabs(double x)
    double sqrt(double x)
    double floor(double x)
    double acos(double x)

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

from libc.time cimport time
from libc.stdlib cimport srand, rand



def img2d_to_ijv_features(double[:, ::1] img2d):

    cdef:
        cnp.npy_intp i=0, j=0
        cnp.npy_intp I = img2d.shape[0], J = img2d.shape[1]
        cnp.npy_intp cnt = 0
        double[:, ::1] features_np = np.empty((I * J, 3), dtype='f8')
        # double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))
        double * features = &features_np[0][0]

    with nogil:

        for i in range(I):
            for j in range(J):

                    features[cnt * 3 + 0] = i
                    features[cnt * 3 + 1] = j
                    features[cnt * 3 + 2] = img2d[i, j]
                    cnt += 1

    # features_np = np.asarray(<double[:I * J, :3]> features)

    return np.asarray(features_np)

def img2d_to_ijv_color(double[:, :, ::1] img2d):

    cdef:
        cnp.npy_intp i=0, j=0
        cnp.npy_intp I = img2d.shape[0], J = img2d.shape[1]
        cnp.npy_intp cnt = 0
        double[:, ::1] features_np = np.empty((I * J, 5), dtype='f8')
        # double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))
        double * features = &features_np[0][0]

    with nogil:

        for i in range(I):
            for j in range(J):
                    features[cnt * 5 + 0] = i
                    features[cnt * 5 + 1] = j
                    features[cnt * 5 + 2] = img2d[i, j, 0]
                    features[cnt * 5 + 3] = img2d[i, j, 1]
                    features[cnt * 5 + 4] = img2d[i, j, 2]
                    cnt += 1

    # features_np = np.asarray(<double[:I * J, :5]> features)

    return np.asarray(features_np)

def img3d_to_ijkv_features(double[:, :, ::1] img3d):

    cdef:
        cnp.npy_intp i=0, j=0, k=0
        cnp.npy_intp I = img3d.shape[0], J = img3d.shape[1], K = img3d.shape[2]
        cnp.npy_intp cnt = 0
        double[:, ::1] features_np = np.empty((I * J * K, 4), dtype='f8')
        # double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))
        double * features = &features_np[0][0]

    with nogil:

        for i in range(I):
            for j in range(J):
                for k in range(K):

                        features[cnt * 4 + 0] = i
                        features[cnt * 4 + 1] = j
                        features[cnt * 4 + 2] = k
                        features[cnt * 4 + 3] = img3d[i, j, k]
                        cnt += 1

    # features_np = np.asarray(<double[:I * J * K, :4]> features)

    return np.asarray(features_np)


# def img3d_to_ijkv_features_free(double[:, ::1] img3d):

#     cdef double * features = &img3d[0][0]

#     with nogil:
#         free(features)


def img3d_integrals(double[:, :, ::1] img3d, double[:, :, ::1] out, cnp.npy_intp radius=20):

    cdef:
        cnp.npy_intp i=0, j=0, k=0, l=0, m=0, n=0
        cnp.npy_intp I = img3d.shape[0], J = img3d.shape[1], K = img3d.shape[2]
        cnp.npy_intp cnt = 0
        cnp.npy_intp sz = radius

    with nogil:

        for i in range(sz, I - sz):
            for j in range(sz, J - sz):
                for k in range(sz, K - sz):

                        for l in range(-sz, sz + 1):
                            for m in range(-sz, sz + 1):
                                for n in range(-sz, sz + 1):

                                    out[i, j, k] += img3d[i + l, j + m, k + n]

    return 1


