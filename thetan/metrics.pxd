# theta.pxd
cimport numpy as cnp

# Declare the function pointer type
ctypedef double (*metricp)(double*, double*, cnp.npy_intp) noexcept nogil

# Declare the cdef function signature
cdef metricp get_metric_func(str metric) except NULL