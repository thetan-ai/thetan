# cython: cdivision=True

cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, acos, atan2, isfinite, NAN

cdef double euclidean(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Euclidean distance between two vectors.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    sqrt(summ) : float
        Square root of the sum of squared differences.
    """
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return sqrt(summ)


cdef double inf_norm(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Chebyshev (infinity norm) distance between two vectors.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    overall_max : float
        Largest absolute difference between the two vectors. NaN if any
        of the differences is NaN, as with the other metrics.
    """
    cdef:
        cnp.npy_intp d
        double overall_max = 0.0
        double total = 0.0
        double tmp_max

    for d in range(D):
        tmp_max = fabs(ptr1[d] - ptr2[d])
        total += tmp_max
        if tmp_max > overall_max:
            overall_max = tmp_max

    # tmp_max > overall_max is False for NaN, so a NaN difference never
    # becomes the max. The running total is NaN if any difference was, and
    # checking it here keeps the loop free of extra branches.
    if total != total:
        return total

    return overall_max


cdef double norm_ssd(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Mean squared difference between two vectors.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    summ / D : float
        Sum of squared differences divided by D. NaN if D is 0.
    """
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ / D


cdef double ssd(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Sum of squared differences between two vectors.

    This is the squared Euclidean distance.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    summ : float
        Sum of squared differences.
    """
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += (ptr1[d] - ptr2[d]) * (ptr1[d] - ptr2[d])

    return summ


cdef double dot_prod(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Dot product of two vectors.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    summ : float
        Sum of the products.
    """
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += ptr1[d] * ptr2[d]

    return summ

cdef double neg_dot_prod(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Negative dot product of two vectors.

    This is what metric='dot_prod' uses. Flipping the sign makes the most
    similar vectors give the smallest value, which is what sequential and
    berserker treat as nearest. A point joins a cluster when its dot
    product with the centroid is greater than -theta, so theta is usually
    negative: theta=-0.9 means join when the dot product is above 0.9.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    -summ : float
        Dot product with its sign flipped.
    """
    # same loop as dot_prod, written out here because calling dot_prod
    # made this noticeably slower on short vectors
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += ptr1[d] * ptr2[d]

    return -summ

cdef double magnitude(double * vec, cnp.npy_intp D) noexcept nogil:
    """Euclidean length of a vector.

    Parameters
    ----------
    vec : double *
        Pointer to the vector.
    D : npy_intp
        Number of values in the vector.

    Returns
    -------
    sqrt(mag) : float
        Square root of the sum of squared values.
    """
    # magnitude of vector of size D
    cdef:
        double mag = 0.0
        cnp.npy_intp d

    for d in range(D):
        mag += vec[d] * vec[d]

    return sqrt(mag)

cdef inline bint safe_length(double mag) noexcept nogil:
    """Check that a length from magnitude can be used as it is.

    Parameters
    ----------
    mag : double
        Length returned by magnitude.

    Returns
    -------
    1e-140 <= mag <= 1e140 : bool
        False if mag is zero, NaN, infinite, or so large or small that the
        squares it was computed from overflowed, underflowed or lost
        accuracy. cosine_distance and arccosine_distance then use
        scaled_angular_distance.
    """
    return 1e-140 <= mag <= 1e140

cdef double max_abs(double * vec, cnp.npy_intp D) noexcept nogil:
    """Largest absolute value in a vector.

    Parameters
    ----------
    vec : double *
        Pointer to the vector.
    D : npy_intp
        Number of values in the vector.

    Returns
    -------
    overall_max : float
        Largest absolute value. NaN if any value is NaN.
    """
    cdef:
        cnp.npy_intp d
        double overall_max = 0.0
        double total = 0.0
        double tmp

    for d in range(D):
        tmp = fabs(vec[d])
        total += tmp
        if tmp > overall_max:
            overall_max = tmp

    # the total is NaN if any value was (see inf_norm)
    if total != total:
        return total

    return overall_max

cdef double scaled_angular_distance(double * vec1, double * vec2, cnp.npy_intp D, bint as_angle) noexcept nogil:
    """Cosine or arccos distance for vectors with zero or extreme lengths.

    cosine_distance and arccosine_distance call this when safe_length is
    False for either vector. Each vector is divided by its largest
    absolute value before anything is squared. That does not change its
    direction and keeps the squares in range, so the result does not
    depend on how long the vectors are.

    Parameters
    ----------
    vec1 : double *
        Pointer to the first vector.
    vec2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.
    as_angle : bool
        If True return the angle, as arccosine_distance does. If False
        return 1 minus the cosine similarity, as cosine_distance does.

    Returns
    -------
    2.0 * atan2(sqrt(diff_sq), sqrt(sum_sq)) or dist : float
        The distance. Returns 0 if both vectors have zero length. If only
        one has zero length returns pi/2 when as_angle is True and 1
        otherwise. Returns NaN if either vector has a NaN or infinite
        value.
    """
    cdef:
        cnp.npy_intp d
        double m1, m2, n1, n2, u, v, dist
        double ss1 = 0.0
        double ss2 = 0.0
        double diff_sq = 0.0
        double sum_sq = 0.0

    m1 = max_abs(vec1, D)
    m2 = max_abs(vec2, D)

    if not (isfinite(m1) and isfinite(m2)):
        return NAN

    if m1 == 0 and m2 == 0:
        return 0.0

    if m1 == 0 or m2 == 0:
        if as_angle:
            return acos(0.0)
        return 1.0

    # lengths of the rescaled vectors, each between 1 and sqrt(D)
    for d in range(D):
        u = vec1[d] / m1
        v = vec2[d] / m2
        ss1 += u * u
        ss2 += v * v

    n1 = sqrt(ss1)
    n2 = sqrt(ss2)

    for d in range(D):
        u = vec1[d] / m1 / n1
        v = vec2[d] / m2 / n2
        diff_sq += (u - v) * (u - v)
        sum_sq += (u + v) * (u + v)

    if as_angle:
        return 2.0 * atan2(sqrt(diff_sq), sqrt(sum_sq))

    dist = 0.5 * diff_sq

    if dist > 2.0:
        dist = 2.0

    return dist

cdef double cosine_distance(double * vec1, double *vec2, cnp.npy_intp D) noexcept nogil:
    """Cosine distance between two vectors.

    For most pairs this is 1 minus the cosine similarity. For nearly
    parallel vectors (similarity above 0.999) that subtraction loses
    accuracy and returns about 1e-16 instead of 0 for identical vectors,
    so the distance is computed from the two unit vectors u and v as
    |u - v|^2 / 2, which is the same quantity.

    Parameters
    ----------
    vec1 : double *
        Pointer to the first vector.
    vec2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    1.0 - cosine_similarity or 0.5 * diff_sq : float
        1 minus the cosine similarity, between 0 and 2. Returns 1 if
        only one vector has zero length, treating it as having a cosine
        similarity of 0 with every other vector. Returns 0 if both have
        zero length, so zero vectors end up in one cluster. Vectors with
        very large or very small values are handled by
        scaled_angular_distance, so the result does not depend on their
        length.
    """

    cdef:
        cnp.npy_intp d
        double dot, mag1, mag2, inv1, inv2, u, v, cosine_similarity
        double diff_sq = 0.0

    mag1 = magnitude(vec1, D)
    mag2 = magnitude(vec2, D)

    if not (safe_length(mag1) and safe_length(mag2)):
        return scaled_angular_distance(vec1, vec2, D, False)

    dot = dot_prod(vec1, vec2, D)
    cosine_similarity = dot / (mag1 * mag2)

    if cosine_similarity <= 0.999:
        if cosine_similarity < -1.0:
            cosine_similarity = -1.0
        return 1.0 - cosine_similarity

    # nearly parallel: 1 - cosine_similarity would lose most of its digits
    inv1 = 1.0 / mag1
    inv2 = 1.0 / mag2

    for d in range(D):
        u = vec1[d] * inv1
        v = vec2[d] * inv2
        diff_sq += (u - v) * (u - v)

    return 0.5 * diff_sq

cdef double arccosine_distance(double * vec1, double *vec2, cnp.npy_intp D) noexcept nogil:
    """Angle between two vectors.

    For most pairs this is acos of the cosine similarity. For nearly
    parallel or nearly opposite vectors (similarity above 0.999 or below
    -0.999) acos loses accuracy and returns about 2e-8 instead of 0 for
    identical vectors, so the angle is computed from the two unit vectors
    u and v as 2 * atan2(|u - v|, |u + v|), which is the same angle.

    Parameters
    ----------
    vec1 : double *
        Pointer to the first vector.
    vec2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    acos(cosine_similarity) or 2.0 * atan2(sqrt(diff_sq), sqrt(sum_sq)) : float
        Angle in radians, between 0 and pi. Returns pi/2 if only one
        vector has zero length, treating it as having a cosine similarity
        of 0 with every other vector. Returns 0 if both have zero length,
        so zero vectors end up in one cluster. Vectors with very large or
        very small values are handled by scaled_angular_distance, so the
        result does not depend on their length.
    """

    cdef:
        cnp.npy_intp d
        double dot, mag1, mag2, inv1, inv2, u, v, cosine_similarity
        double diff_sq = 0.0
        double sum_sq = 0.0

    mag1 = magnitude(vec1, D)
    mag2 = magnitude(vec2, D)

    if not (safe_length(mag1) and safe_length(mag2)):
        return scaled_angular_distance(vec1, vec2, D, True)

    dot = dot_prod(vec1, vec2, D)
    cosine_similarity = dot / (mag1 * mag2)

    if -0.999 <= cosine_similarity <= 0.999:
        return acos(cosine_similarity)

    # nearly parallel or opposite: acos would lose most of its digits
    inv1 = 1.0 / mag1
    inv2 = 1.0 / mag2

    for d in range(D):
        u = vec1[d] * inv1
        v = vec2[d] * inv2
        diff_sq += (u - v) * (u - v)
        sum_sq += (u + v) * (u + v)

    return 2.0 * atan2(sqrt(diff_sq), sqrt(sum_sq))


cdef double l1_norm(double * ptr1, double * ptr2, cnp.npy_intp D) noexcept nogil:
    """Manhattan (L1) distance between two vectors.

    Parameters
    ----------
    ptr1 : double *
        Pointer to the first vector.
    ptr2 : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    summ : float
        Sum of absolute differences.
    """
    cdef:
        cnp.npy_intp d
        double summ = 0.0

    for d in range(D):
        summ += fabs(ptr1[d] - ptr2[d])

    return summ


cdef double norm_ssd_un(double * a, double * b, cnp.npy_intp D) noexcept nogil:
    """Mean squared difference between two vectors.

    Same result as norm_ssd, but it reads four values at a time to speed
    up the loop.

    Parameters
    ----------
    a : double *
        Pointer to the first vector.
    b : double *
        Pointer to the second vector.
    D : npy_intp
        Number of values in each vector.

    Returns
    -------
    result / D : float
        Sum of squared differences divided by D. NaN if D is 0.
    """
    cdef:
        cnp.npy_intp i
        cnp.npy_intp n = D // 4
        cnp.npy_intp rem = D % 4
        double result = 0

    for i in range(n):
        result += ((a[0] - b[0]) * (a[0] - b[0])
                  +(a[1] - b[1]) * (a[1] - b[1])
                  +(a[2] - b[2]) * (a[2] - b[2])
                  +(a[3] - b[3]) * (a[3] - b[3]))
        a += 4; b += 4

    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result / D

cdef metricp get_metric_func(object metric) except NULL:
    """Look up a metric function by name.

    Parameters
    ----------
    metric : str
        One of 'euclidean', 'ssd', 'norm_ssd', 'norm_ssd_un', 'dot_prod',
        'l1_norm', 'inf_norm', 'cosine' or 'arccos'. 'dot_prod' gives the
        negative dot product (see neg_dot_prod). Subclasses of str, such
        as numpy strings and string enums, are accepted.

    Returns
    -------
    &euclidean, &ssd, ..., &arccosine_distance : metricp
        Pointer to the matching metric function.

    Raises
    ------
    TypeError
        If metric is not a str.
    ValueError
        If metric is not one of the names above.
    """
    if not isinstance(metric, str):
        raise TypeError(
            f"metric must be a str, got {type(metric).__name__}."
        )

    if metric == 'euclidean':
        return &euclidean
    if metric == 'ssd':
        return &ssd
    if metric == 'norm_ssd':
        return &norm_ssd
    if metric == 'norm_ssd_un':
        return &norm_ssd_un
    if metric == 'dot_prod':
        return &neg_dot_prod
    if metric == 'l1_norm':
        return &l1_norm
    if metric == 'inf_norm':
        return &inf_norm
    if metric == 'cosine':
        return &cosine_distance
    if metric == 'arccos':
        return &arccosine_distance
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose one of 'euclidean', 'ssd', "
            f"'norm_ssd', 'norm_ssd_un', 'dot_prod', 'l1_norm', 'inf_norm', "
            f"'cosine' or 'arccos'."
        )

def distance_matrix(double[:, ::1] X1, double[:, ::1] X2, metric='norm_ssd'):
    """Compute the metric between every row of X1 and every row of X2.

    Parameters
    ----------
    X1 : ndarray, shape (n1, n_features)
        First set of vectors, as a C-contiguous float64 array.
    X2 : ndarray, shape (n2, n_features)
        Second set of vectors, as a C-contiguous float64 array with the
        same number of columns as X1.
    metric : str, optional
        One of 'euclidean', 'ssd', 'norm_ssd', 'norm_ssd_un', 'dot_prod',
        'l1_norm', 'inf_norm', 'cosine' or 'arccos'. Default is
        'norm_ssd', not 'euclidean'. 'dot_prod' gives the negative dot
        product, so that for every metric a smaller value means the rows
        are more alike.

    Returns
    -------
    np.asarray(distance_matrix) : ndarray, shape (n1, n2)
        Value of the metric for each pair of rows. With 'dot_prod' these
        are the dot products with their sign flipped.

    Raises
    ------
    TypeError
        If metric is not a str.
    ValueError
        If X1 and X2 have different numbers of columns, or metric is not a
        known name.
    """

    cdef:
        cnp.npy_intp i = 0
        cnp.npy_intp j = 0
        cnp.npy_intp I = X1.shape[0]
        cnp.npy_intp J = X2.shape[0]
        cnp.npy_intp D = X1.shape[1]
        metricp metric_func = &euclidean
        double[:, ::1] distance_matrix

    if X2.shape[1] != D:
        raise ValueError(
            f"X1 and X2 must have the same number of columns, "
            f"got {D} and {X2.shape[1]}."
        )

    distance_matrix = np.zeros((I, J), dtype=np.float64)

    metric_func = get_metric_func(metric)

    for i in range(I):
        for j in range(J):
            distance_matrix[i, j] = metric_func(&X1[i, 0], &X2[j, 0], D)

    return np.asarray(distance_matrix)
