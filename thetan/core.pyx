# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp
# from scipy.linalg.cython_blas cimport ddot
from cython.parallel cimport prange
from thetan.hashing import hash32, hash64, hash_rows_32, hash_rows_64

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

ctypedef double (*metricp)(double *, double *, cnp.npy_intp) noexcept nogil

ctypedef struct Linkedlist:

    cnp.npy_intp value
    double dist
    Linkedlist * next


ctypedef struct Node:
    cnp.npy_intp sz
    cnp.npy_intp * elements






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


# // Function to calculate the magnitude of a vector
# double magnitude(double *vec, int n) {
#     double mag = 0.0;
#     for (int i = 0; i < n; i++) {
#         mag += vec[i] * vec[i];
#     }
#     return sqrt(mag);
# }

cdef double magnitude(double * vec, cnp.npy_intp D) noexcept nogil:
    # magnitude of vector of size D
    cdef:
        double mag = 0.0
        cnp.npy_intp d

    for d in range(D):
        mag += vec[d] * vec[d]

    return sqrt(mag)


# // Function to calculate the cosine distance
# double cosine_distance(double *vec1, double *vec2, int n) {
#     double dot = dot_product(vec1, vec2, n);
#     double mag1 = magnitude(vec1, n);
#     double mag2 = magnitude(vec2, n);

#     if (mag1 == 0 || mag2 == 0) {
#         fprintf(stderr, "Error: One or both vectors have zero magnitude.\n");
#         return -1; // Return error value
#     }

#     double cosine_similarity = dot / (mag1 * mag2);
#     return 1.0 - cosine_similarity; // Convert similarity to distance
# }

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


# // Function to calculate the arccosine distance
# double arccosine_distance(double *vec1, double *vec2, int n) {
#     double dot = dot_product(vec1, vec2, n);
#     double mag1 = magnitude(vec1, n);
#     double mag2 = magnitude(vec2, n);

#     if (mag1 == 0 || mag2 == 0) {
#         fprintf(stderr, "Error: One or both vectors have zero magnitude.\n");
#         return -1; // Return error value
#     }

#     double cosine_similarity = dot / (mag1 * mag2);

#     // Clamp the cosine similarity to the range [-1, 1] to handle numerical inaccuracies
#     if (cosine_similarity < -1.0) cosine_similarity = -1.0;
#     if (cosine_similarity > 1.0) cosine_similarity = 1.0;

#     return acos(cosine_similarity); // Return arccosine distance
# }


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


cdef void swap(cnp.npy_intp * a, cnp.npy_intp * b) noexcept nogil:
    cdef cnp.npy_intp temp = a[0]
    a[0] = b[0]
    b[0] = temp


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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

    for i in range(I):
        for j in range(J):
            distance_matrix[i, j] = metric_func(&X1[i, 0], &X2[j, 0], D)

    return np.asarray(distance_matrix)


def sequential(double[:, ::1] X, double theta, metric='euclidean', all_out=False, verbose=False):

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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

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


def sequential_pexact(double[:, ::1] X, double theta, metric='euclidean', all_out=False, verbose=False):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0, d = 0
        cnp.npy_intp k = 0, K = 0
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double * summs = <double *> malloc(<size_t> D * sizeof(double))
        double * centroids = <double *> malloc(<size_t> D * sizeof(double))
        double * clusters_sizes = <double *> malloc(<size_t> 1 * sizeof(double))
        double * distances = <double *> malloc(<size_t> N * sizeof(double))  # workspace for distances
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean
        double cnt_orig = 0

    # choose metric function under the GIL
    if metric == 'euclidean':
        metric_func = &euclidean
    if metric == 'ssd':
        metric_func = &ssd
    if metric == 'norm_ssd':
        metric_func = &norm_ssd
    if metric == 'norm_ssd_un':
        metric_func = &norm_ssd_un
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

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

            # --- PARALLELIZED PART: compute all distances in parallel over k ---
            for k in prange(K, schedule='static'):
                distances[k] = metric_func(&X[i, 0], &centroids[k * D], D)

            # same number of metric calls as before
            cnt_orig += K

            # --- SERIAL argmin, to keep tie-breaking identical ---
            for k in range(K):
                dist = distances[k]
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

    if verbose:
        print('Original calls ', cnt_orig)

    final_centroids = np.asarray(<double[:K, :D]> centroids)
    final_indices = np.asarray(indices)

    # distances is only a temporary buffer, always free it
    free(distances)

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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

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


    # TODO: in case none of the labels are negative this will return an empty list
    reassigned_indices = np.asarray(indices)

    return reassigned_indices


def sequential_prelim(double[:, ::1] X, double[:, ::1] means_prev, double theta, metric='euclidean', all_out=False):

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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

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


def centroids_from_labels(const double[:, ::1] X, cnp.npy_intp[:] indices):

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
    # cluster_sizes = NULL
    # final_cluster_sizes = np.asarray(<double[:K]> cluster_sizes, dtype=np.float64)
    final_centroids = np.asarray(<double[:K, :D]> centroids, dtype=np.float64)
    # del final_cluster_sizes
    return np.ascontiguousarray(final_centroids)


def wcss(double[:, ::1] X, cnp.npy_intp[:] indices, double[:, ::1] centroids, metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        double summ = 0, dist = 0
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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance


    with nogil:

        for i in range(N):
            dist = metric_func(&X[i, 0], &centroids[indices[i], 0], D)
            summ += dist ** 2

    return summ


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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

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


def assign_pexact(double[:, ::1] X, double[:, ::1] centroids, metric='euclidean'):

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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

    with nogil:

        for i in prange(N, schedule='static'):

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


def assign_hashx(double[:, ::1] X, double[:, ::1] centroids, double theta, metric='euclidean'):

    cdef:
        cnp.npy_intp N = X.shape[0]
        cnp.npy_intp D = X.shape[1]
        cnp.npy_intp i = 0
        cnp.npy_intp k = 0, K = centroids.shape[0]
        cnp.npy_intp [:] indices = np.empty(N, dtype='i8')
        double dist = 0, minimum = DBL_MAX
        cnp.npy_intp argmin = -1
        metricp metric_func = &euclidean
        cnp.uint64_t [:] row_Xhashes
        cnp.uint64_t [:] row_chashes


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
    if metric == 'dot_prod':
        metric_func = &dot_prod
    if metric == 'l1_norm':
        metric_func = &l1_norm
    if metric == 'cosine':
        metric_func = &cosine_distance
    if metric == 'arccos':
        metric_func = &arccosine_distance

    Xarr = np.ascontiguousarray(np.round(np.asarray(X) / (1.5 * theta), 0), dtype=np.uint64)
    row_Xhashes = hash_rows_64(Xarr)

    Carr = np.ascontiguousarray(np.round(np.asarray(centroids) / (1.5 * theta), 0), dtype=np.uint64)
    row_chashes = hash_rows_64(Carr)

    with nogil:

        for i in range(N):

            minimum = DBL_MAX
            argmin = -1

            for k in range(K):

                if row_Xhashes[i] != row_chashes[k]:
                    continue

                dist = metric_func(&X[i, 0], &centroids[k, 0], D)

                if dist < minimum:
                    minimum = dist
                    argmin = k

            indices[i] = argmin

    final_indices = np.asarray(indices)

    return final_indices