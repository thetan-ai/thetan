# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt, acos, fabs, log, exp

# from cythonutils cimport same_shape
from featurespeed cimport IdentityFeature, ResampleFeature

DEF biggest_double = 1.7976931348623157e+308  #  np.finfo('f8').max


import math
cdef double PI = math.pi


cdef class Metric(object):
    """ Computes a distance between two sequential data.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions). A `feature` object can be specified
    in order to calculate the distance between extracted features, rather
    than directly between the sequential data.

    Parameters
    ----------
    feature : `Feature` object, optional
        It is used to extract features before computing the distance.

    Notes
    -----
    When subclassing `Metric`, one only needs to override the `dist` and
    `are_compatible` methods.
    """
    def __init__(Metric self, Feature feature=IdentityFeature()):
        self.feature = feature

    property feature:
        """ `Feature` object used to extract features from sequential data """
        def __get__(Metric self):
            return self.feature

    cdef double c_dist(Metric self, Data2D features1, Data2D features2) nogil except -1:
        """ Cython version of `Metric.dist`. """
        with gil:
            _features1 = np.asarray(<float[:features1.shape[0], :features1.shape[1]]> <float*> features1._data)
            _features2 = np.asarray(<float[:features2.shape[0], :features2.shape[1]]> <float*> features2._data)
            return self.dist(_features1, _features2)

    cpdef double dist(Metric self, features1, features2) except -1:
        """ Computes a distance between two data points based on their features.

        Parameters
        ----------
        features1 : 2D array
            Features of the first data point.
        features2 : 2D array
            Features of the second data point.

        Returns
        -------
        double
            Distance between two data points.
        """
        raise NotImplementedError("Metric's subclasses must implement method `dist(self, features1, features2)`!")


cdef class CythonMetric(Metric):
    """ Computes a distance between two sequential data.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions). A `feature` object can be specified
    in order to calculate the distance between extracted features, rather
    than directly between the sequential data.

    Parameters
    ----------
    feature : `Feature` object, optional
        It is used to extract features before computing the distance.

    Notes
    -----
    When subclassing `CythonMetric`, one only needs to override the `c_dist` and
    `c_are_compatible` methods.
    """

    cpdef double dist(CythonMetric self, features1, features2) except -1:
        """ Computes a distance between two data points based on their features.

        Parameters
        ----------
        features1 : 2D array
            Features of the first data point.
        features2 : 2D array
            Features of the second data point.

        Returns
        -------
        double
            Distance between two data points.

        Notes
        -----
        This method calls its Cython version `self.c_dist` accordingly.
        """
        # If needed, we convert features to 2D arrays.
        features1 = np.asarray(features1)
        if features1.ndim == 0:
            features1 = features1[np.newaxis, np.newaxis]
        elif features1.ndim == 1:
            features1 = features1[np.newaxis]
        elif features1.ndim == 2:
            pass
        else:
            raise TypeError("Only scalar, 1D or 2D array features are"
                            " supported for parameter 'features1'!")

        features2 = np.asarray(features2)
        if features2.ndim == 0:
            features2 = features2[np.newaxis, np.newaxis]
        elif features2.ndim == 1:
            features2 = features2[np.newaxis]
        elif features2.ndim == 2:
            pass
        else:
            raise TypeError("Only scalar, 1D or 2D array features are"
                            " supported for parameter 'features2'!")

        return self.c_dist(features1, features2)


cdef class SumPointwiseEuclideanMetric(CythonMetric):
    r""" Computes the sum of pointwise Euclidean distances between two
    sequential data.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions). A `feature` object can be specified
    in order to calculate the distance between the features, rather than
    directly between the sequential data.

    Parameters
    ----------
    feature : `Feature` object, optional
        It is used to extract features before computing the distance.

    Notes
    -----
    The distance between two 2D sequential data::

        s1       s2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $a+b+c$ where $a$ is the Euclidean distance between s1[0] and
    s2[0], $b$ between s1[1] and s2[1] and $c$ between s1[2] and s2[2].
    """
    cdef double c_dist(SumPointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef :
            int N = features1.shape[0], D = features1.shape[1]
            int n, d
            double dd, dist_n, dist = 0.0

        for n in range(N):
            dist_n = 0.0
            for d in range(D):
                dd = features1[n, d] - features2[n, d]
                dist_n += dd * dd

            dist += sqrt(dist_n)

        return dist


cdef class AveragePointwiseEuclideanMetric(SumPointwiseEuclideanMetric):
    r""" Computes the average of pointwise Euclidean distances between two
    sequential data.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions). A `feature` object can be specified
    in order to calculate the distance between the features, rather than
    directly between the sequential data.

    Parameters
    ----------
    feature : `Feature` object, optional
        It is used to extract features before computing the distance.

    Notes
    -----
    The distance between two 2D sequential data::

        s1       s2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $(a+b+c)/3$ where $a$ is the Euclidean distance between s1[0] and
    s2[0], $b$ between s1[1] and s2[1] and $c$ between s1[2] and s2[2].
    """
    cdef double c_dist(AveragePointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef int N = features1.shape[0]
        cdef double dist = SumPointwiseEuclideanMetric.c_dist(self, features1, features2)
        return dist / N


cdef class CosineMetric(CythonMetric):
    r""" Computes the cosine distance between two vectors.

    A vector (i.e. a N-dimensional point) is represented as a 2D array with
    shape (1, nb_dimensions).

    Notes
    -----
    The distance between two vectors $v_1$ and $v_2$ is equal to
    $\frac{1}{\pi} \arccos\left(\frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}\right)$
    and is bounded within $[0,1]$.
    """
    def __init__(CosineMetric self, Feature feature):
        super(CosineMetric, self).__init__(feature=feature)

    cdef double c_dist(CosineMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef :
            int d, D = features1.shape[1]
            double sqr_norm_features1 = 0.0, sqr_norm_features2 = 0.0
            double cos_theta = 0.0

        for d in range(D):
            cos_theta += features1[0, d] * features2[0, d]
            sqr_norm_features1 += features1[0, d] * features1[0, d]
            sqr_norm_features2 += features2[0, d] * features2[0, d]

        if sqr_norm_features1 == 0.:
            if sqr_norm_features2 == 0.:
                return 0.
            else:
                return 1.

        cos_theta /= sqrt(sqr_norm_features1) * sqrt(sqr_norm_features2)

        # Make sure it's in [-1, 1], i.e. within domain of arccosine
        cos_theta = min(cos_theta, 1.)
        cos_theta = max(cos_theta, -1.)
        return acos(cos_theta) / PI  # Normalized cosine distance


cdef class CrossCorrelationMetric(CythonMetric):
    r""" Computes the cross correlation distance between two vectors.

    A vector (i.e. a N-dimensional point) is represented as a 2D array with
    shape (1, nb_dimensions).

    """
    def __init__(CrossCorrelationMetric self, Feature feature):
        super(CrossCorrelationMetric, self).__init__(feature=feature)

    cdef double c_dist(CrossCorrelationMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef :
            int d, D = features1.shape[1]
            double sqr_norm_features1 = 0.0, sqr_norm_features2 = 0.0
            double cc = 0.0
            double dot_product = 0.0

        for d in range(D):
            dot_product += features1[0, d] * features2[0, d]
            sqr_norm_features1 += features1[0, d] * features1[0, d]
            sqr_norm_features2 += features2[0, d] * features2[0, d]

        if sqr_norm_features1 == 0.:
            if sqr_norm_features2 == 0.:
                return 0.
            else:
                return 1.

        cc = dot_product / (sqrt(sqr_norm_features1) * sqrt(sqr_norm_features2))

        return 1 - fabs(cc)
        # Make sure it's in [-1, 1], i.e. within domain of arccosine
        #cos_theta = min(cos_theta, 1.)
        #cos_theta = max(cos_theta, -1.)
        #return acos(cos_theta) / PI  # Normalized cosine distance


cdef class KLDivergenceMetric(CythonMetric):
    r""" Computes the KL divergence between two vectors.

    A vector (i.e. a N-dimensional point) is represented as a 2D array with
    shape (1, nb_dimensions).

    Due to the nature of the metric, each feature
    should be positive values

    """
    def __init__(KLDivergenceMetric self, Feature feature):
        super(KLDivergenceMetric, self).__init__(feature=feature)

    cdef double c_dist(KLDivergenceMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef :
            int d, D = features1.shape[1]
            int size = features1.shape[0] * features1.shape[1]
            double sz = <double> size
            double kl = 0.0

        for d in range(D):
            if features1[0, d] != 0 and features2[0, d] != 0:
                kl += features1[0, d] * log(features1[0, d]/features2[0, d])

        return kl / sz

cdef class NormalKLDivergenceMetric(CythonMetric):
    r""" Computes the normalized KL divergence between two vectors.

    A vector (i.e. a N-dimensional point) is represented as a 2D array with
    shape (1, nb_dimensions).

    Due to the nature of the metric, each feature
    should be positive values

    """
    def __init__(NormalKLDivergenceMetric self, Feature feature):
        super(NormalKLDivergenceMetric, self).__init__(feature=feature)

    cdef double c_dist(NormalKLDivergenceMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef :
            int d, D = features1.shape[1]
            int size = features1.shape[0] * features1.shape[1]
            double sz = <double> size
            double kl = 0.0

        for d in range(D):
            if features1[0, d] != 0 and features2[0, d] != 0:
                kl += features1[0, d] * log(features1[0, d]/features2[0, d])

        kl = 1 - exp(kl / sz)
        return kl


cdef class NormalizedSumSquaredDiffMetric(CythonMetric):
    r""" Computes the normalized SSD distance between two features.

    """
    def __init__(NormalizedSumSquaredDiffMetric self, Feature feature):
        super(NormalizedSumSquaredDiffMetric, self).__init__(feature=feature)

    cdef double c_dist(NormalizedSumSquaredDiffMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef:
            int s
            int size = features1.shape[0] * features1.shape[1]
            double sz = <double> size
            float * ptr1 = &features1[0, 0]
            float * ptr2 = &features2[0, 0]
            double summ = 0.0


        for s in range(size):
            summ += (ptr1[s] - ptr2[s]) * (ptr1[s] - ptr2[s])

        return summ / sz

    cdef double c_distp(NormalizedSumSquaredDiffMetric self, float * ptr1, float * ptr2, int size) nogil:
            cdef:
                int s
                # int size = features1.shape[0] * features1.shape[1]
                double sz = <double> size
                # float * ptr1 = &features1[0, 0]
                # float * ptr2 = &features2[0, 0]
                double summ = 0.0


            for s in range(size):
                summ += (ptr1[s] - ptr2[s]) * (ptr1[s] - ptr2[s])

            return summ / sz

cdef class NormalizedSumSquaredRootDiffMetric(CythonMetric):
    r""" Computes the normalized SSD distance between two features.

    """
    def __init__(NormalizedSumSquaredRootDiffMetric self, Feature feature):
        super(NormalizedSumSquaredRootDiffMetric, self).__init__(feature=feature)

    cdef double c_dist(NormalizedSumSquaredRootDiffMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef:
            int s
            int size = features1.shape[0] * features1.shape[1]
            double sz = <double> size
            float * ptr1 = &features1[0, 0]
            float * ptr2 = &features2[0, 0]
            double summ = 0.0


        for s in range(size):
            summ += sqrt((ptr1[s] - ptr2[s]) * (ptr1[s] - ptr2[s]))

        return summ / sz

    cdef double c_distp(NormalizedSumSquaredRootDiffMetric self, float * ptr1, float * ptr2, int size) nogil:
            cdef:
                int s
                # int size = features1.shape[0] * features1.shape[1]
                double sz = <double> size
                # float * ptr1 = &features1[0, 0]
                # float * ptr2 = &features2[0, 0]
                double summ = 0.0


            for s in range(size):
                summ += sqrt((ptr1[s] - ptr2[s]) * (ptr1[s] - ptr2[s]))

            return summ / sz


cdef class NormalizedSumAbsDiffMetric(CythonMetric):
    r""" Computes the normalized ASD distance between two features.

    """
    def __init__(NormalizedSumAbsDiffMetric self, Feature feature):
        super(NormalizedSumAbsDiffMetric, self).__init__(feature=feature)

    cdef double c_dist(NormalizedSumAbsDiffMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef:
            int s
            int size = features1.shape[0] * features1.shape[1]
            double sz = <double> size
            float * ptr1 = &features1[0, 0]
            float * ptr2 = &features2[0, 0]
            double summ = 0.0


        for s in range(size):
            summ += fabs(ptr1[s] - ptr2[s])

        return summ / sz

cdef class NormalizedSumInfDiffMetric(CythonMetric):
    r""" Computes the normalized ASD distance between two features.

    """
    def __init__(NormalizedSumInfDiffMetric self, Feature feature):
        super(NormalizedSumInfDiffMetric, self).__init__(feature=feature)

    cdef double c_dist(NormalizedSumInfDiffMetric self, Data2D features1, Data2D features2) nogil except -1:
        cdef:
            int s
            int size = features1.shape[0] * features1.shape[1]
            double sz = <double> size
            float * ptr1 = &features1[0, 0]
            float * ptr2 = &features2[0, 0]
            double temp
            double max_v = 0.0


        for s in range(size):
            temp = fabs(ptr1[s] - ptr2[s])
            if temp > max_v:
                max_v = temp

        return max_v / sz

cpdef distance_matrix(Metric metric, data1, data2=None):
    """ Computes the distance matrix between two lists of sequential data.

    The distance matrix is obtained by computing the pairwise distance of all
    tuples spawn by the Cartesian product of `data1` with `data2`. If `data2`
    is not provided, the Cartesian product of `data1` with itself is used
    instead. A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    Parameters
    ----------
    metric : `Metric` object
        Tells how to compute the distance between two sequential data.
    data1 : list of 2D arrays
        List of sequences of N-dimensional points.
    data2 : list of 2D arrays
        List of sequences of N-dimensional points.

    Returns
    -------
    2D array (double)
        Distance matrix.
    """
    cdef int i, j
    if data2 is None:
        data2 = data1

    # TODO this should be called size?
    shape = data1[0].shape
    distance_matrix = np.zeros((len(data1), len(data2)), dtype=np.float64)
    cdef:
        Data2D features1 = np.empty(shape, np.float32)
        Data2D features2 = np.empty(shape, np.float32)

    for i in range(len(data1)):
        datum1 = data1[i] if data1[i].flags.writeable and data1[i].dtype is np.float32 else data1[i].astype(np.float32)
        metric.feature.c_extract(datum1, features1)
        for j in range(len(data2)):
            datum2 = data2[j] if data2[j].flags.writeable and data2[j].dtype is np.float32 else data2[j].astype(np.float32)
            metric.feature.c_extract(datum2, features2)
            distance_matrix[i, j] = metric.c_dist(features1, features2)

    return distance_matrix


cpdef assign_to_centroid(Metric metric, data, centroids):
    """ Assigns data points to their closest centroids

    Parameters
    ----------
    metric : `Metric` object
        Tells how to compute the distance between two sequential data.
    data : list of 2D arrays
        List of sequences of N-dimensional points.
    centroid : list of 2D arrays
        List of sequences of N-dimensional points.

    Returns
    -------
    labels : int array
    """
    cdef cnp.npy_intp i, j

    # TODO this should be called size?
    shape = data[0].shape

    cdef:
        cnp.npy_intp [::1] labels = np.zeros((len(data)), dtype=np.intp)
        double dist = 0.0
        double curr_min = biggest_double
        cnp.npy_intp curr_id = -1
        Data2D features1 = np.empty(shape, np.float32)
        Data2D features2 = np.empty(shape, np.float32)

    for i in range(len(data)):
        datum1 = data[i] if data[i].flags.writeable and data[i].dtype is np.float32 else data[i].astype(np.float32)
        metric.feature.c_extract(datum1, features1)

        curr_min = biggest_double
        for j in range(len(centroids)):
            datum2 = centroids[j] if centroids[j].flags.writeable and centroids[j].dtype is np.float32 else centroids[j].astype(np.float32)
            metric.feature.c_extract(datum2, features2)
            dist = metric.c_dist(features1, features2)
            if dist < curr_min:
                curr_min = dist
                curr_id = j
        labels[i] = curr_id

    return np.asarray(labels)


cdef double c_distp_l2(float * ptr1, float * ptr2, int size) nogil:
        cdef:
            int s
            double summ = 0.0

        for s in range(size):
            summ += sqrt((ptr1[s] - ptr2[s]) * (ptr1[s] - ptr2[s]))

        return summ


cdef double c_distp_nl2(float * ptr1, float * ptr2, int size) nogil:
        cdef:
            int s
            double sz = <double> size
            double summ = 0.0

        for s in range(size):
            summ += sqrt((ptr1[s] - ptr2[s]) * (ptr1[s] - ptr2[s]))

        return summ / sz


cdef double c_distp_nl1(float * ptr1, float * ptr2, int size) nogil:
        cdef:
            int s
            double sz = <double> size
            double summ = 0.0

        for s in range(size):
            summ += fabs(ptr1[s] - ptr2[s])

        return summ / sz

cdef double c_distp_nlinf(float * ptr1, float * ptr2, int size) nogil:
        cdef:
            int s
            double sz = <double> size
            double max_v = 0.0
            double temp        

        for s in range(size):
            temp = fabs(ptr1[s] - ptr2[s])
            if temp > max_v:
                max_v = temp
        
        return max_v / sz

cdef double c_distp_kldiv(float * ptr1, float * ptr2, int size) nogil:
        cdef:
            int s
            double sz = <double> size
            double kl = 0.0
        
        for s in range(size):
            if ptr1[s] > 0 and ptr2[s] > 0:
                kl += ptr1[s] * log(ptr1[s]/ptr2[s])

        return kl / sz

cdef double c_distp_normkl(float * ptr1, float * ptr2, int size) nogil:
        cdef:
            int s
            double sz = <double> size
            double kl = 0.0

        for s in range(size):
            if ptr1[s] > 0 and ptr2[s] > 0:
                kl += ptr1[s] * log(ptr1[s]/ptr2[s])

        return 1 - exp(kl / sz)





def assign_to_centroid_nl2(Metric metric, float [:, :, ::1] data, float [:, :, ::1] centroids):

    cdef:
        cnp.npy_intp i, j
        cnp.npy_intp [::1] labels = np.zeros((len(data)), dtype=np.intp)
        double dist = 0.0
        double curr_min = biggest_double
        cnp.npy_intp curr_id = -1
        cnp.npy_intp I = <cnp.npy_intp> data.shape[0]
        cnp.npy_intp J = <cnp.npy_intp> centroids.shape[0]
        int size = data.shape[1] * data.shape[2]

    with nogil:

        for i in range(I):
            curr_min = biggest_double
            for j in range(J):
                # dist = metric.c_dist(features1, features2)
                dist = c_distp_nl2(&data[i, 0, 0], &centroids[j, 0, 0], size)
                if dist < curr_min:
                    curr_min = dist
                    curr_id = j
            labels[i] = curr_id

    return np.asarray(labels)


def assign_to_centroid_nl1(Metric metric, float [:, :, ::1] data, float [:, :, ::1] centroids):

    cdef:
        cnp.npy_intp i, j
        cnp.npy_intp [::1] labels = np.zeros((len(data)), dtype=np.intp)
        double dist = 0.0
        double curr_min = biggest_double
        cnp.npy_intp curr_id = -1
        cnp.npy_intp I = <cnp.npy_intp> data.shape[0]
        cnp.npy_intp J = <cnp.npy_intp> centroids.shape[0]
        int size = data.shape[1] * data.shape[2]

    with nogil:

        for i in range(I):
            curr_min = biggest_double
            for j in range(J):
                # dist = metric.c_dist(features1, features2)
                dist = c_distp_nl1(&data[i, 0, 0], &centroids[j, 0, 0], size)
                if dist < curr_min:
                    curr_min = dist
                    curr_id = j
            labels[i] = curr_id

    return np.asarray(labels)

def assign_to_centroid_nlinf(Metric metric, float [:, :, ::1] data, float [:, :, ::1] centroids):

    cdef:
        cnp.npy_intp i, j
        cnp.npy_intp [::1] labels = np.zeros((len(data)), dtype=np.intp)
        double dist = 0.0
        double curr_min = biggest_double
        cnp.npy_intp curr_id = -1
        cnp.npy_intp I = <cnp.npy_intp> data.shape[0]
        cnp.npy_intp J = <cnp.npy_intp> centroids.shape[0]
        int size = data.shape[1] * data.shape[2]

    with nogil:

        for i in range(I):
            curr_min = biggest_double
            for j in range(J):
                # dist = metric.c_dist(features1, features2)
                dist = c_distp_nlinf(&data[i, 0, 0], &centroids[j, 0, 0], size)
                if dist < curr_min:
                    curr_min = dist
                    curr_id = j
            labels[i] = curr_id

    return np.asarray(labels)

def assign_to_centroid_kldiv(Metric metric, float [:, :, ::1] data, float [:, :, ::1] centroids):

    cdef:
        cnp.npy_intp i, j
        cnp.npy_intp [::1] labels = np.zeros((len(data)), dtype=np.intp)
        double dist = 0.0
        double curr_min = biggest_double
        cnp.npy_intp curr_id = -1
        cnp.npy_intp I = <cnp.npy_intp> data.shape[0]
        cnp.npy_intp J = <cnp.npy_intp> centroids.shape[0]
        int size = data.shape[1] * data.shape[2]

    with nogil:

        for i in range(I):
            curr_min = biggest_double
            for j in range(J):
                # dist = metric.c_dist(features1, features2)
                dist = c_distp_kldiv(&data[i, 0, 0], &centroids[j, 0, 0], size)
                if dist < curr_min:
                    curr_min = dist
                    curr_id = j
            labels[i] = curr_id

    return np.asarray(labels)

def assign_to_centroid_normkl(Metric metric, float [:, :, ::1] data, float [:, :, ::1] centroids):

    cdef:
        cnp.npy_intp i, j
        cnp.npy_intp [::1] labels = np.zeros((len(data)), dtype=np.intp)
        double dist = 0.0
        double curr_min = biggest_double
        cnp.npy_intp curr_id = -1
        cnp.npy_intp I = <cnp.npy_intp> data.shape[0]
        cnp.npy_intp J = <cnp.npy_intp> centroids.shape[0]
        int size = data.shape[1] * data.shape[2]

    with nogil:

        for i in range(I):
            curr_min = biggest_double
            for j in range(J):
                # dist = metric.c_dist(features1, features2)
                dist = c_distp_normkl(&data[i, 0, 0], &centroids[j, 0, 0], size)
                if dist < curr_min:
                    curr_min = dist
                    curr_id = j
            labels[i] = curr_id

    return np.asarray(labels)

cpdef double dist(Metric metric, datum1, datum2) except -1:
    """ Computes a distance between `datum1` and `datum2`.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    Parameters
    ----------
    metric : `Metric` object
        Tells how to compute the distance between `datum1` and `datum2`.
    datum1 : 2D array
        Sequence of N-dimensional points.
    datum2 : 2D array
        Sequence of N-dimensional points.

    Returns
    -------
    double
        Distance between two data points.
    """
    datum1 = datum1 if datum1.flags.writeable and datum1.dtype is np.float32 else datum1.astype(np.float32)
    datum2 = datum2 if datum2.flags.writeable and datum2.dtype is np.float32 else datum2.astype(np.float32)

    cdef:
        #Shape shape1 = metric.feature.c_infer_shape(datum1)
        #Shape shape2 = metric.feature.c_infer_shape(datum2)
        Data2D features1 = np.empty(datum1.shape, np.float32)
        Data2D features2 = np.empty(datum2.shape, np.float32)

    metric.feature.c_extract(datum1, features1)
    metric.feature.c_extract(datum2, features2)
    return metric.c_dist(features1, features2)


def patch_distance_matrix(float [:, ::1] f1,
                          float [:, ::1] f2,
                          cnp.npy_intp f1_size,
                          cnp.npy_intp f2_size,
                          cnp.npy_intp cols,
                          double [:, ::1] D):
    """ Experimental distance matrix
    """

    cdef:
        cnp.npy_intp i=0, j=0, mov_i=0, mov_j=0

    with nogil:

        for i in range(f1_size):
            for j in range(f2_size):

                D[i, j] = c_distp_l2(&f1[i, 0],
                                     &f2[j, 0],
                                     cols)

    return np.asarray(D)
