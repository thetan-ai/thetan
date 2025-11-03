import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free, calloc
from libc.string cimport memcpy

cnp.import_array()

DBSCAN_NOISE = -1
DBSCAN_UNCLASSIFIED = -2

cdef int C_DBSCAN_NOISE = -1
cdef int C_DBSCAN_UNCLASSIFIED = -2

# double precision
ctypedef cnp.float64_t DTYPE_t 

# for labels
ctypedef cnp.int32_t ITYPE_t 

# Dynamic Int Vector
cdef struct IntVec:
    int* data
    int size
    int cap

cdef inline int iv_init(IntVec* v, int cap_hint) nogil:
    if cap_hint <=0:
        cap_hint = 16
    v.size = 0
    v.cap = cap_hint
    v.data = <int*> malloc(<size_t>v.cap * sizeof(int))
    return 0 if v.data != NULL else -1

cdef inline void iv_free(IntVec* v) noexcept nogil:
    if v.data != NULL:
        free(v.data)
    v.data = NULL
    v.size = 0
    v.cap = 0

cdef inline void iv_clear(IntVec* v) noexcept nogil:
    v.size = 0

cdef inline int iv_push(IntVec* v, int x) nogil:
    cdef int new_cap
    cdef int* nd
    
    if v.size == v.cap:
        if v.cap < 65536:
            new_cap = v.cap * 2
        else:
            new_cap = v.cap + v.cap // 2

        nd = <int*> realloc(v.data, <size_t>new_cap * sizeof(int))
        if nd == NULL:
            return -1
        v.data = nd
        v.cap = new_cap
    
    v.data[v.size] = x
    v.size += 1
    return 0

#Squared Euclidean Distance
cdef inline double dist2_rowmajor(const double* X, int dim, int i, int j) nogil:
    cdef const double* a = X + (<size_t>i) * dim
    cdef const double* b = X + (<size_t>j) * dim
    cdef int k
    cdef double s = 0.0
    cdef double d
    for k in range(dim):
        d = a[k] - b[k]
        s += d * d
    return s

# Region Query
cdef int region_query(const double* X, int n, int dim, int idx, double eps2, IntVec* out) nogil:
    cdef int j
    cdef double d2
    iv_clear(out)
    for j in range(n):
        d2 = dist2_rowmajor(X, dim, idx, j)
        if d2 <= eps2:
            if iv_push(out, j) != 0:
                return -1
    return 0

# Cluster Expansion
cdef int expand_cluster(const double* X, int n, int dim,
                        double eps2, int min_pts,
                        int* labels, unsigned char* visited,
                        int cid, const IntVec* seeds_in) nogil:
    cdef IntVec queue
    cdef int i, q, t
    cdef IntVec qnbrs
    cdef int qpos

    if iv_init(&queue, seeds_in.size if seeds_in.size > 0 else 16) != 0:
        return -1
    
    for i in range(seeds_in.size):
        if iv_push(&queue, seeds_in.data[i]) != 0:
            iv_free(&queue)
            return -1
    
    qpos = 0
    while qpos < queue.size:
        q = queue.data[qpos]
        qpos += 1

        if visited[q] == 0:
            visited[q] = 1
        
            if iv_init(&qnbrs, 32) != 0:
                iv_free(&queue)
                return -1
            
            if region_query(X, n, dim, q, eps2, &qnbrs) != 0:
                iv_free(&qnbrs)
                iv_free(&queue)
                return -1
            
            if qnbrs.size >= min_pts:
                for t in range(qnbrs.size):
                    if iv_push(&queue, qnbrs.data[t]) != 0:
                        iv_free(&qnbrs)
                        iv_free(&queue)
                        return -1
            iv_free(&qnbrs)
    
        if labels[q] == C_DBSCAN_UNCLASSIFIED or labels[q] == C_DBSCAN_NOISE:
            labels[q] = cid
    iv_free(&queue)
    return 0

cdef int _dbscan_core(const double* X, int n_points, int dim, double eps, int min_pts, int* labels) nogil:
    cdef int i, p
    cdef double eps2 = eps * eps

    if X == NULL or labels == NULL or n_points <= 0 or dim <= 0 or eps <= 0.0:
        for i in range(n_points):
            labels[i] = C_DBSCAN_NOISE
        return 0
    
    if min_pts < 1:
        min_pts = 1
    
    # For visited flags
    cdef unsigned char* visited = <unsigned char*> calloc(<size_t>n_points, sizeof(unsigned char))
    if visited == NULL:
        for i in range(n_points):
            labels[i] = C_DBSCAN_NOISE
        return 0

    # Init Labels
    for i in range(n_points):
        labels[i] = C_DBSCAN_UNCLASSIFIED
    
    cdef IntVec nbrs
    if iv_init(&nbrs, 64) != 0:
        free(visited)
        for i in range(n_points):
            labels[i] = C_DBSCAN_NOISE
        return 0
    
    cdef int cid = 0
    for p in range(n_points):
        if visited[p] != 0:
            continue
        visited[p] = 1
    
        if region_query(X, n_points, dim, p, eps2, &nbrs) != 0:
            iv_free(&nbrs)
            free(visited)
            for i in range(n_points):
                labels[i] = C_DBSCAN_NOISE
            return 0
        
        if nbrs.size < min_pts:
            labels[p] = C_DBSCAN_NOISE
            continue
        
        labels[p] = cid
        if expand_cluster(X, n_points, dim, eps2, min_pts, labels, visited, cid, &nbrs) != 0:
            iv_free(&nbrs)
            free(visited)
            for i in range(n_points):
                labels[i] = C_DBSCAN_NOISE
            return 0
        cid += 1
    
    iv_free(&nbrs)
    free(visited)
    return cid

# Python Facing API
def dbscan(X, double eps, int min_pts = 5):
    # Ensuring contiguous float64 array
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] Xc = np.ascontiguousarray(X, dtype=np.float64)
    cdef Py_ssize_t n_points = Xc.shape[0]
    cdef Py_ssize_t dim = Xc.shape[1]

    # For output Labels
    cdef cnp.ndarray[ITYPE_t, ndim=1, mode='c'] labels_out = np.empty(n_points, dtype=np.int32) 

    cdef int* labels_c = <int*> malloc(<size_t>n_points * sizeof(int))
    if labels_c == NULL:
        raise MemoryError("Failed to allocate labels buffer")
    
    cdef int n_clusters 
    with nogil:
        n_clusters = _dbscan_core(<const double*> Xc.data, <int>n_points, <int>dim, eps, min_pts, labels_c)

    # Copying back into Numpy array
    memcpy(<void*>labels_out.data, <void*>labels_c, <size_t>n_points * sizeof(int))
    free(labels_c)

    return labels_out, n_clusters