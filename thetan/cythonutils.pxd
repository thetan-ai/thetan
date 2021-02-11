# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

cimport numpy as cnp

cdef extern from "cythonutils.h":
    enum: MAX_NDIM

ctypedef float[:] Data1D
ctypedef float[:,:] Data2D


ctypedef fused Data:
    Data1D
    Data2D


cdef struct Shape:
   Py_ssize_t ndim
   Py_ssize_t dims[MAX_NDIM]
   Py_ssize_t size

cdef Shape shape_from_memview(Data data) nogil

cdef Data2D* create_memview_2d(Py_ssize_t buffer_size, Py_ssize_t dims[MAX_NDIM]) nogil

cdef void free_memview_2d(Data2D* memview) nogil
