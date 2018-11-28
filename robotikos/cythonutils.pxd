# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

cimport numpy as cnp

cdef extern from "cythonutils.h":
    enum: MAX_NDIM

ctypedef float[:] Data1D
ctypedef float[:,:] Data2D
ctypedef float[:,:,:] Data3D
ctypedef float[:,:,:,:] Data4D
ctypedef float[:,:,:,:,:] Data5D
ctypedef float[:,:,:,:,:,:] Data6D
ctypedef float[:,:,:,:,:,:,:] Data7D

ctypedef fused Data:
    Data1D
    Data2D
    Data3D
    Data4D
    Data5D
    Data6D
    Data7D

cdef struct Shape:
   Py_ssize_t ndim
   Py_ssize_t dims[MAX_NDIM]
   Py_ssize_t size

cdef Shape shape_from_memview(Data data) nogil

""" TODO completely remove these functions

cdef Shape tuple2shape_buggy(dims) except *


cdef shape2tuple_buggy(Shape shape)


cdef void tuple2shape(cnp.npy_intp [:] dims, Shape *shape)


cdef void shape2tuple(Shape *shape, cnp.npy_intp [:] dims)


cdef int same_shape(Shape shape1, Shape shape2) nogil
"""

cdef Data2D* create_memview_2d(Py_ssize_t buffer_size, Py_ssize_t dims[MAX_NDIM]) nogil

cdef void free_memview_2d(Data2D* memview) nogil
