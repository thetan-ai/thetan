from cythonutils cimport Data2D, Shape
from featurespeed cimport Feature


cdef class Metric(object):
    cdef Feature feature
    cdef int is_order_invariant

    cdef double c_dist(Metric self, Data2D features1, Data2D features2) nogil except -1

    cpdef double dist(Metric self, features1, features2) except -1
