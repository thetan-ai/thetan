# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True

"""
Cython wrapper for xxHash functions.
"""
import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t

cdef extern from "xxhash.h":
    uint32_t XXH32(const void* data, size_t length, uint32_t seed)
    uint64_t XXH64(const void* data, size_t length, uint64_t seed)

def hash32(bytes data, uint32_t seed=0):
    """Compute XXH32 hash of data with an optional seed.

    Parameters
    ----------
    data : bytes
        Input data to hash
    seed : uint32_t, optional
        A seed to alter the hash result predictably, by default 0

    Returns
    -------
    uint32_t
        The 32-bit hash value
    """
    if not data:
        return XXH32(NULL, 0, seed)
    return XXH32(<const void*>data, len(data), seed)

def hash64(bytes data, uint64_t seed=0):
    """Compute XXH64 hash of data with an optional seed.

    Parameters
    ----------
    data : bytes
        Input data to hash
    seed : uint64_t, optional
        A seed to alter the hash result predictably, by default 0

    Returns
    -------
    uint64_t
        The 64-bit hash value
    """
    if not data:
        return XXH64(NULL, 0, seed)
    return XXH64(<const void*>data, len(data), seed)


def hash_rows(cnp.ndarray[uint32_t, ndim=2, cast=True, mode="c"] X, uint32_t seed=0):
    """Compute XXH32 hash of each row in a 2D array with an optional seed.

    Parameters
    ----------
    X : cnp.ndarray
        Input 2D array to hash
    seed : uint32_t, optional
        A seed to alter the hash result predictably, by default 0

    Returns
    -------
    np.uint32_t[:]
        An array of 32-bit hash values for each row

    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef cnp.uint32_t[:] hashes = np.empty(N, dtype=np.uint32)
    cdef int i
    cdef const void* row_ptr
    cdef size_t row_size

    for i in range(N):
        # Get direct pointer to the row data
        row_ptr = <const void*>&X[i, 0]
        # Size in bytes is number of elements * size of element
        row_size = D * sizeof(uint32_t)
        # Hash the row data directly
        hashes[i] = XXH32(row_ptr, row_size, seed)

    return np.asarray(hashes)



