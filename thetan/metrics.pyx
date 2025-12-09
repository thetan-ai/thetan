cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, acos
from libc.float cimport DBL_MIN

cdef double euclidean(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return sqrt(summ)


cdef double inf_norm(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    cdef:
        cnp.npy_intp d
        double overall_max = DBL_MIN
        double tmp_max

    for d in range(D):
        tmp_max = fabs(ptr1[d] - ptr2[d])
        if tmp_max > overall_max:
            overall_max = tmp_max

    return overall_max


cdef double norm_ssd(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ / D


cdef double ssd(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ


cdef double dot_prod(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += ptr1[d] * ptr2[d]

    return summ

cdef double magnitude(double * vec, cnp.npy_intp D) noexcept nogil:
    # magnitude of vector of size D
    cdef:
        double mag = 0.0
        cnp.npy_intp d

    for d in range(D):
        mag += vec[d] * vec[d]

    return sqrt(mag)

cdef double cosine_distance(double * vec1, double *vec2, cnp.npy_intp D) noexcept nogil:

    cdef:
        double dot, mag1, mag2, cosine_similarity

    dot = dot_prod(vec1, vec2, D)
    mag1 = magnitude(vec1, D)
    mag2 = magnitude(vec2, D)

    if mag1 == 0 or mag2 == 0:
        return 1

    cosine_similarity = dot / (mag1 * mag2)

    if cosine_similarity < -1.0:
        cosine_similarity = -1.0

    if cosine_similarity > 1.0:
        cosine_similarity = 1.0

    return 1.0 - cosine_similarity

cdef double arccosine_distance(double * vec1, double *vec2, cnp.npy_intp D) noexcept nogil:

    cdef:
        double dot, mag1, mag2

    dot = dot_prod(vec1, vec2, D)
    mag1 = magnitude(vec1, D)
    mag2 = magnitude(vec2, D)

    if mag1 == 0 or mag2 == 0:
        return 0

    cosine_similarity = dot / (mag1 * mag2)

    if cosine_similarity < -1.0:
        cosine_similarity = -1.0

    if cosine_similarity > 1.0:
        cosine_similarity = 1.0

    return acos(cosine_similarity)


cdef double l1_norm(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += fabs(ptr1[d] - ptr2[d])

    return summ


cdef double norm_ssd_un(double * a, double * b, cnp.npy_intp D) noexcept nogil:
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

cdef metricp get_metric_func(str metric) except NULL:
    """Return the appropriate metric function pointer based on the metric name."""
    if metric == 'euclidean':
        return &euclidean
    if metric == 'ssd':
        return &ssd
    if metric == 'norm_ssd':
        return &norm_ssd
    if metric == 'norm_ssd_un':
        return &norm_ssd_un
    if metric == 'dot_prod':
        return &dot_prod
    if metric == 'l1_norm':
        return &l1_norm
    if metric == 'cosine':
        return &cosine_distance
    if metric == 'arccos':
        return &arccosine_distance
    else:
        raise ValueError(f"Unknown metric.")

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

    metric_func = get_metric_func(metric)

    for i in range(I):
        for j in range(J):
            distance_matrix[i, j] = metric_func(&X1[i, 0], &X2[j, 0], D)

    return np.asarray(distance_matrix)