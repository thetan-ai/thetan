# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as cnp

from cythonutils cimport shape_from_memview
from robotikos.streamlinespeed cimport c_set_number_of_points, c_length


cdef class Feature(object):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    Notes
    -----
    When subclassing `Feature`, one only needs to override the `extract` and
    `infer_shape` methods.
    """
    def __init__(Feature self):

        pass

    cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil except *:
        """ Cython version of `Feature.extract`. """
        cdef Data2D c_features
        with gil:

            features = np.asarray(self.extract(np.asarray(datum))).astype(np.float32)
            # print('Inside features', features)
            # print('features.ndim', features.ndim)
            if features.ndim == 0:
                features = features[np.newaxis, np.newaxis]
            elif features.ndim == 1:
                features = features[np.newaxis]
            elif features.ndim == 2:
                pass
            else:
                raise TypeError("Only scalar, 1D or 2D array features are supported!")
            # print('Outside features', features)
            c_features = features
            # print('After c assignment')

        # out[:] = c_features
        # This needs real testing
        out = c_features

        # with gil:
        #    # print('After out assignment')


    cpdef extract(Feature self, datum):
        """ Extracts features from a sequential datum.

        Parameters
        ----------
        datum : 2D array
            Sequence of N-dimensional points.

        Returns
        -------
        2D array
            Features extracted from `datum`.
        """
        raise NotImplementedError("Feature's subclasses must implement method `extract(self, datum)`!")


cdef class CythonFeature(Feature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).


    Notes
    -----
    By default, when inheriting from `CythonFeature`, Python methods will call their
    C version (e.g. `CythonFeature.extract` -> `self.c_extract`).
    """

    cpdef extract(CythonFeature self, datum):
        """ Extracts features from a sequential datum.

        Parameters
        ----------
        datum : 2D array
            Sequence of N-dimensional points.

        Returns
        -------
        2D array
            Features extracted from `datum`.

        Notes
        -----
        This method calls its Cython version `self.c_extract` accordingly.
        """
        if not datum.flags.writeable or datum.dtype is not np.float32:
            datum = datum.astype(np.float32)

        # shape = shape2tuple(self.c_infer_shape(datum))
        # TODO rename this to size
        shape = datum.shape
        cdef Data2D out = np.empty(shape, dtype=datum.dtype)
        self.c_extract(datum, out)
        return np.asarray(out)


cdef class IdentityFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The features being extracted are the actual sequence's points. This is
    useful for metric that does not require any pre-processing.
    """
    def __init__(IdentityFeature self):
        super(IdentityFeature, self).__init__()

    cdef void c_extract(IdentityFeature self, Data2D datum, Data2D out) nogil except *:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int n, d

        for n in range(N):
            for d in range(D):
                out[n, d] = datum[n, d]


cdef class ResampleFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The features being extracted are the points of the sequence once resampled.
    This is useful for metrics requiring a constant number of points for all
     streamlines.
    """
    def __init__(ResampleFeature self, cnp.npy_intp nb_points):
        super(ResampleFeature, self).__init__()
        self.nb_points = nb_points

        if nb_points <= 0:
            raise ValueError("ResampleFeature: `nb_points` must be strictly positive: {0}".format(nb_points))
    """
    cdef Shape c_infer_shape(ResampleFeature self, Data2D datum) nogil except *:
        cdef Shape shape = shape_from_memview(datum)
        shape.dims[0] = self.nb_points
        return shape
    """
    cdef void c_extract(ResampleFeature self, Data2D datum, Data2D out) nogil except *:
        c_set_number_of_points(datum, out)


cdef class CenterOfMassFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists of one N-dimensional point representing
    the mean of the points, i.e. the center of mass.
    """
    def __init__(CenterOfMassFeature self):
        super(CenterOfMassFeature, self).__init__()

    cdef void c_extract(CenterOfMassFeature self, Data2D datum, Data2D out) nogil except *:
        cdef int N = datum.shape[0], D = datum.shape[1]
        cdef int i, d

        for d in range(D):
            out[0, d] = 0

        for i in range(N):
            for d in range(D):
                out[0, d] += datum[i, d]

        for d in range(D):
            out[0, d] /= N


cdef class MidpointFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists of one N-dimensional point representing
    the middle point of the sequence (i.e. `nb_points//2`th point).
    """
    def __init__(MidpointFeature self):
        super(MidpointFeature, self).__init__()

    cdef void c_extract(MidpointFeature self, Data2D datum, Data2D out) nogil except *:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int mid = N/2
            int d

        for d in range(D):
            out[0, d] = datum[mid, d]

