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

cdef extern from "xxhash.h" nogil:
    uint32_t XXH32(const void* data, size_t length, uint32_t seed)
    uint64_t XXH64(const void* data, size_t length, uint64_t seed)

# from cython.parallel cimport prange


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


def hash_rows_32(cnp.ndarray[uint32_t, ndim=2, cast=True, mode="c"] X, uint32_t seed=0):
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

    with nogil:

        for i in range(N):
            # Get direct pointer to the row data
            row_ptr = <const void*>&X[i, 0]
            # Size in bytes is number of elements * size of element
            row_size = D * sizeof(uint32_t)
            # Hash the row data directly
            hashes[i] = XXH32(row_ptr, row_size, seed)

    return np.asarray(hashes)


def hash_rows_64(cnp.ndarray[uint64_t, ndim=2, cast=True, mode="c"] X, uint64_t seed=0):
    """Compute XXH64 hash of each row in a 2D array with an optional seed.

    Parameters
    ----------
    X : cnp.ndarray
        Input 2D array to hash
    seed : uint64_t, optional
        A seed to alter the hash result predictably, by default 0

    Returns
    -------
    np.uint64_t[:]
        An array of 64-bit hash values for each row

    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef cnp.uint64_t[:] hashes = np.empty(N, dtype=np.uint64)
    cdef int i
    cdef const void* row_ptr
    cdef size_t row_size

    with nogil:

        for i in range(N):
            # Get direct pointer to the row data
            row_ptr = <const void*>&X[i, 0]
            # Size in bytes is number of elements * size of element
            row_size = D * sizeof(uint64_t)
            # Hash the row data directly
            hashes[i] = XXH64(row_ptr, row_size, seed)

    return np.asarray(hashes)



def simhash_xxhash_64(double[:, ::1] X, double[:, ::1] R, uint64_t seed=0):
    """
    Compute SimHash (cosine LSH) + XXH64 for rows of X.

    Parameters
    ----------
    X : (N, D) double, normalized
    R : (K, D) double, random hyperplanes
    seed : uint64

    Returns
    -------
    hashes : np.uint64 (N,)
    """

    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t D = X.shape[1]
        Py_ssize_t K = R.shape[0]

        Py_ssize_t i, j, d
        Py_ssize_t nwords = (K + 63) // 64

        cnp.ndarray[uint64_t, ndim=2] bits = np.zeros((N, nwords), dtype=np.uint64)
        cnp.ndarray[uint64_t, ndim=1] hashes = np.empty(N, dtype=np.uint64)

        double dot = 0.0
        uint64_t word = 0
        uint64_t mask = 0

    with nogil:
        for i in range(N):

            # compute sign(R x)
            for j in range(K):

                dot = 0.0
                for d in range(D):
                    dot += R[j, d] * X[i, d]

                if dot > 0:
                    word = j >> 6          # j // 64
                    mask = <uint64_t>1 << (j & 63)
                    bits[i, word] |= mask

            # hash packed bits
            hashes[i] = XXH64(
                <const void*>&bits[i, 0],
                nwords * sizeof(uint64_t),
                seed
            )

    return np.asarray(hashes)
