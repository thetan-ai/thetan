import numpy as np
cimport numpy as cnp
from libc.math cimport fabs
from libc.stdlib cimport malloc, free

cnp.import_array()

def find_neighbor_pairs(
    cnp.int64_t[:, :, :, :, :, :] label_patches,
    cnp.float64_t[:, :, :, :, :, :] data_patches,
    double var_threshold=1e-6,
):
    cdef Py_ssize_t n0 = label_patches.shape[0]
    cdef Py_ssize_t n1 = label_patches.shape[1]
    cdef Py_ssize_t n2 = label_patches.shape[2]
    cdef Py_ssize_t p0 = label_patches.shape[3]
    cdef Py_ssize_t p1 = label_patches.shape[4]
    cdef Py_ssize_t p2 = label_patches.shape[5]
    cdef Py_ssize_t patch_n = p0 * p1 * p2

    cdef Py_ssize_t i, j, k, pi, pj, pk, a_idx, b_idx
    cdef int skipped = 0
    cdef double mean_val, var_val, v
    cdef cnp.int64_t la, lb, pair_min, pair_max

    cdef cnp.int64_t *flat_labels = <cnp.int64_t *>malloc(patch_n * sizeof(cnp.int64_t))
    cdef cnp.int64_t *unique_buf = <cnp.int64_t *>malloc(patch_n * sizeof(cnp.int64_t))
    cdef Py_ssize_t n_unique, idx, already

    neighbor_pairs = set()

    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                # This part is for checking the variance of the patch. If it's too low, we skip it.
                mean_val = 0.0
                idx = 0
                for pi in range(p0):
                    for pj in range(p1):
                        for pk in range(p2):
                            v = data_patches[i, j, k, pi, pj, pk]
                            mean_val += v
                            idx += 1
                mean_val /= patch_n

                var_val = 0.0
                for pi in range(p0):
                    for pj in range(p1):
                        for pk in range(p2):
                            v = data_patches[i, j, k, pi, pj, pk] - mean_val
                            var_val += v * v
                var_val /= patch_n

                if var_val < var_threshold:
                    skipped += 1
                    continue

                n_unique = 0
                for pi in range(p0):
                    for pj in range(p1):
                        for pk in range(p2):
                            la = label_patches[i, j, k, pi, pj, pk]
                            already = 0
                            for idx in range(n_unique):
                                if unique_buf[idx] == la:
                                    already = 1
                                    break
                            if already == 0:
                                unique_buf[n_unique] = la
                                n_unique += 1
                if n_unique < 2:
                    continue
                

                for a_idx in range(n_unique):
                    for b_idx in range(a_idx + 1, n_unique):
                        la = unique_buf[a_idx]
                        lb = unique_buf[b_idx]
                        if la < lb:
                            pair_min = la
                            pair_max = lb
                        else:
                            pair_min = lb
                            pair_max = la
                        neighbor_pairs.add((pair_min, pair_max))
    
    free(flat_labels)
    free(unique_buf)

    return neighbor_pairs, skipped

# Union-Find 

cdef class UnionFind:
    cdef cnp.int64_t[:] _parent
    cdef cnp.int64_t[:] _sz
    cdef cnp.float64_t[:] _min_int
    cdef cnp.float64_t[:] _max_int
    cdef Py_ssize_t _n

    def __cinit__(self, Py_ssize_t n,
                 cnp.int64_t[:] sizes,
                 cnp.float64_t[:] centroid_ints):
        
        cdef Py_ssize_t i
        self._n = n
        self._parent = np.arange(n, dtype=np.int64)
        self._sz = np.array(sizes, dtype=np.int64)
        self._min_int = np.array(centroid_ints, dtype=np.float64)
        self._max_int = np.array(centroid_ints, dtype=np.float64)

    cdef Py_ssize_t _find(self, Py_ssize_t x) noexcept nogil:
        cdef Py_ssize_t root = x
        while self._parent[root] != root:
            root = self._parent[root]
        cdef Py_ssize_t tmp
        while self._parent[x] != root:
            tmp = self._parent[x]
            self._parent[x] = root
            x = tmp
        return root
    
    cpdef Py_ssize_t find(self, Py_ssize_t x):
        return self._find(x)
    
    cpdef bint union(self, Py_ssize_t x, Py_ssize_t y,
                     double centroid_x, double centroid_y,
                     double max_intensity_threshold):
        
        cdef Py_ssize_t rx = self._find(x)
        cdef Py_ssize_t ry = self._find(y)
        if rx == ry:
            return True
        
        cdef double new_min, new_max
        if max_intensity_threshold > 0:
            new_min = self._min_int[rx]
            if self._min_int[ry] < new_min:
                new_min = self._min_int[ry]
            
            new_max = self._max_int[rx]
            if self._max_int[ry] > new_max:
                new_max = self._max_int[ry]
            if centroid_x > new_max:
                new_max = centroid_x
            if centroid_y > new_max:
                new_max = centroid_y
            
            if new_max - new_min > max_intensity_threshold:
                return False
        
        else:
            new_min = self._min_int[rx]
            if self._min_int[ry] < new_min:
                new_min = self._min_int[ry]
            new_max = self._max_int[rx]
            if self._max_int[ry] > new_max:
                new_max = self._max_int[ry]
            
        if self._sz[rx] >= self._sz[ry]:
            self._parent[ry] = rx
            self._sz[rx] += self._sz[ry]
            self._min_int[rx] = new_min
            self._max_int[rx] = new_max
        else:
            self._parent[rx] = ry
            self._sz[ry] += self._sz[rx]
            self._min_int[ry] = new_min
            self._max_int[ry] = new_max
        return True
    

def compute_cluster_stats(
    cnp.int64_t[:,:,:] labels,
    cnp.float64_t[:,:,:] data,
    Py_ssize_t n_labels,
):
    cdef Py_ssize_t n0 = labels.shape[0]
    cdef Py_ssize_t n1 = labels.shape[1]
    cdef Py_ssize_t n2 = labels.shape[2]
    cdef Py_ssize_t i, j, k
    cdef cnp.int64_t lab

    cdef cnp.float64_t[:] sums = np.zeros(n_labels, dtype=np.float64)
    cdef cnp.int64_t[:] counts = np.zeros(n_labels, dtype=np.int64)

    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                lab = labels[i, j, k]
                sums[lab] += data[i, j, k]
                counts[lab] += 1
    
    cdef cnp.float64_t[:] means = np.zeros(n_labels, dtype=np.float64)
    for i in range(n_labels):
        if counts[i] > 0:
            means[i] = sums[i] / counts[i]
    
    return np.asarray(means), np.asarray(counts)


def remap_labels(
    cnp.int64_t[:,:,:] labels,
    cnp.int64_t[:] label_map,
):
    cdef Py_ssize_t d0 = labels.shape[0]
    cdef Py_ssize_t d1 = labels.shape[1]
    cdef Py_ssize_t d2 = labels.shape[2]
    cdef Py_ssize_t i, j, k

    for i in range(d0):
        for j in range(d1):
            for k in range(d2):
                labels[i, j, k] = label_map[labels[i, j, k]]