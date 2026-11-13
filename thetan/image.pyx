# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

cimport cython
import numpy as np
cimport numpy as cnp
# from scipy.linalg.cython_blas cimport ddot


from libc.math cimport fabs
cdef extern from "math.h" nogil:
    double fabs(double x)
    double sqrt(double x)
    double floor(double x)
    double acos(double x)

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t elsize)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

cdef extern from "limits.h":
    cdef int INT_MAX
    cdef unsigned int UINT_MAX

cdef extern from "float.h":
    cdef double DBL_MAX
    cdef double DBL_MIN

from libc.time cimport time
from libc.stdlib cimport srand, rand



def img2d_to_ijv_features(double[:, ::1] img2d):

    cdef:
        cnp.npy_intp i=0, j=0
        cnp.npy_intp I = img2d.shape[0], J = img2d.shape[1]
        cnp.npy_intp cnt = 0
        double[:, ::1] features_np = np.empty((I * J, 3), dtype='f8')
        # double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))
        double * features = &features_np[0][0]

    with nogil:

        for i in range(I):
            for j in range(J):

                    features[cnt * 3 + 0] = i
                    features[cnt * 3 + 1] = j
                    features[cnt * 3 + 2] = img2d[i, j]
                    cnt += 1

    # features_np = np.asarray(<double[:I * J, :3]> features)

    return np.asarray(features_np)

def img2d_to_ijv_color(double[:, :, ::1] img2d):

    cdef:
        cnp.npy_intp i=0, j=0
        cnp.npy_intp I = img2d.shape[0], J = img2d.shape[1]
        cnp.npy_intp cnt = 0
        double[:, ::1] features_np = np.empty((I * J, 5), dtype='f8')
        # double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))
        double * features = &features_np[0][0]

    with nogil:

        for i in range(I):
            for j in range(J):
                    features[cnt * 5 + 0] = i
                    features[cnt * 5 + 1] = j
                    features[cnt * 5 + 2] = img2d[i, j, 0]
                    features[cnt * 5 + 3] = img2d[i, j, 1]
                    features[cnt * 5 + 4] = img2d[i, j, 2]
                    cnt += 1

    # features_np = np.asarray(<double[:I * J, :5]> features)

    return np.asarray(features_np)

def img3d_to_ijkv_features(double[:, :, ::1] img3d):

    cdef:
        cnp.npy_intp i=0, j=0, k=0
        cnp.npy_intp I = img3d.shape[0], J = img3d.shape[1], K = img3d.shape[2]
        cnp.npy_intp cnt = 0
        double[:, ::1] features_np = np.empty((I * J * K, 4), dtype='f8')
        # double * features = <double *> malloc(<size_t> I * J * K * 4 * sizeof(double))
        double * features = &features_np[0][0]

    with nogil:

        for i in range(I):
            for j in range(J):
                for k in range(K):

                        features[cnt * 4 + 0] = i
                        features[cnt * 4 + 1] = j
                        features[cnt * 4 + 2] = k
                        features[cnt * 4 + 3] = img3d[i, j, k]
                        cnt += 1

    # features_np = np.asarray(<double[:I * J * K, :4]> features)

    return np.asarray(features_np)


# def img3d_to_ijkv_features_free(double[:, ::1] img3d):

#     cdef double * features = &img3d[0][0]

#     with nogil:
#         free(features)


def img3d_integrals(double[:, :, ::1] img3d, double[:, :, ::1] out, cnp.npy_intp radius=20):

    cdef:
        cnp.npy_intp i=0, j=0, k=0, l=0, m=0, n=0
        cnp.npy_intp I = img3d.shape[0], J = img3d.shape[1], K = img3d.shape[2]
        cnp.npy_intp cnt = 0
        cnp.npy_intp sz = radius

    with nogil:

        for i in range(sz, I - sz):
            for j in range(sz, J - sz):
                for k in range(sz, K - sz):

                        for l in range(-sz, sz + 1):
                            for m in range(-sz, sz + 1):
                                for n in range(-sz, sz + 1):

                                    out[i, j, k] += img3d[i + l, j + m, k + n]

    return 1


def correct_connectivity(Py_ssize_t[:, :, ::1] segments,
                         Py_ssize_t min_size,
                         Py_ssize_t max_size,
                         Py_ssize_t start_label=1):
    """ Helper function to remove small disconnected regions from the labels

    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    min_size : int
        The minimum size of the segment
    max_size : int
        The maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    start_label : int
        The label indexing start value.

    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1), dtype=np.intp)

    # new object with connected segments initialized to mask_label
    cdef Py_ssize_t mask_label = start_label - 1

    cdef Py_ssize_t[:, :, ::1] connected_segments \
        = np.full_like(segments, mask_label, dtype=np.intp)

    cdef Py_ssize_t current_new_label = start_label
    cdef Py_ssize_t label = start_label

    # variables for the breadth first search
    cdef Py_ssize_t current_segment_size = 1
    cdef Py_ssize_t bfs_visited = 0
    cdef Py_ssize_t adjacent

    cdef Py_ssize_t zz, yy, xx

    cdef Py_ssize_t[:, ::1] coord_list = np.empty((max_size, 3), dtype=np.intp)

    with nogil:
        for z in range(depth):
            for y in range(height):
                for x in range(width):

                    if segments[z, y, x] == mask_label:
                        continue

                    if connected_segments[z, y, x] > mask_label:
                        continue

                    # find the component size
                    adjacent = current_new_label
                    label = segments[z, y, x]
                    connected_segments[z, y, x] = current_new_label
                    current_segment_size = 1
                    bfs_visited = 0
                    coord_list[bfs_visited, 0] = z
                    coord_list[bfs_visited, 1] = y
                    coord_list[bfs_visited, 2] = x

                    #perform a breadth first search to find
                    # the size of the connected component
                    while bfs_visited < current_segment_size < max_size:
                        for i in range(6):
                            zz = coord_list[bfs_visited, 0] + ddz[i]
                            yy = coord_list[bfs_visited, 1] + ddy[i]
                            xx = coord_list[bfs_visited, 2] + ddx[i]
                            if (0 <= xx < width and
                                0 <= yy < height and
                                0 <= zz < depth):
                                if (segments[zz, yy, xx] == label and
                                    connected_segments[zz, yy, xx] == mask_label):
                                    connected_segments[zz, yy, xx] = \
                                        current_new_label
                                    coord_list[current_segment_size, 0] = zz
                                    coord_list[current_segment_size, 1] = yy
                                    coord_list[current_segment_size, 2] = xx
                                    current_segment_size += 1
                                    if current_segment_size >= max_size:
                                        break
                                elif (connected_segments[zz, yy, xx] > mask_label and
                                      connected_segments[zz, yy, xx] != current_new_label):
                                    adjacent = connected_segments[zz, yy, xx]
                        bfs_visited += 1

                    # change to an adjacent one, like in the original paper
                    if current_segment_size < min_size:
                        for i in range(current_segment_size):
                            connected_segments[coord_list[i, 0],
                                               coord_list[i, 1],
                                               coord_list[i, 2]] = adjacent
                    else:
                        current_new_label += 1

    return np.asarray(connected_segments)



# Union-Find data structure for label merging
cdef inline cnp.npy_intp find_root(cnp.npy_intp *labels, cnp.npy_intp x):
    """Find root of label x using path compression."""
    while labels[x] != x:
        labels[x] = labels[labels[x]]  # Path compression
        x = labels[x]
    return x

cdef inline void union_labels(cnp.npy_intp *labels, cnp.npy_intp a, cnp.npy_intp b):
    """Union two labels into the same component."""
    rootA = find_root(labels, a)
    rootB = find_root(labels, b)
    if rootA != rootB:
        labels[rootB] = rootA  # Merge components


def connected_components(double [:, :, ::1] image, double delta=0.0):
    """
    Perform 3D connected component labeling using 26-connectivity.

    Parameters:
        image (3D np.ndarray): Input image (binary or grayscale).
        delta (float): Intensity difference threshold for merging components.

    Returns:
        labels (3D np.ndarray[intp]): Labeled image with unique components.
        num_components (int): Total number of connected components.
    """
    cdef cnp.npy_intp depth = image.shape[0]
    cdef cnp.npy_intp height = image.shape[1]
    cdef cnp.npy_intp width = image.shape[2]

    # Output labels array
    cdef cnp.npy_intp [:, :, ::1] labels = np.zeros((depth, height, width), dtype=np.intp)

    # Define the 26-connected neighborhood in 3D
    cdef cnp.npy_intp[26][3] neighbors = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),  (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1), (1, -1, -1),
        (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1),
        (1, 1, 0), (1, 1, 1)]


    # Allocate Union-Find structure
    cdef cnp.npy_intp max_labels = depth * height * width + 1  # labels start at 1
    cdef cnp.npy_intp *uf_labels = <cnp.npy_intp*>malloc(max_labels * sizeof(cnp.npy_intp))
    if not uf_labels:
        raise MemoryError("Failed to allocate memory for Union-Find labels")

    # Initialize Union-Find structure
    for i in range(max_labels):
        uf_labels[i] = i

    cdef cnp.npy_intp label_count = 1  # Start labeling from 1
    cdef cnp.npy_intp z, y, x, nz, ny, nx, current_label, neighbor_label

    # First Pass: Assign provisional labels and track equivalences
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if image[z, y, x] > 0:  # Foreground pixel
                    current_label = 0

                    # Check all 26 neighbors
                    for i in range(26):
                        nz, ny, nx = z + neighbors[i][0], y + neighbors[i][1], x + neighbors[i][2]
                        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                            if image[nz, ny, nx] > 0 and abs(image[z, y, x] - image[nz, ny, nx]) <= delta:
                                neighbor_label = labels[nz, ny, nx]
                                if neighbor_label > 0:
                                    if current_label == 0:
                                        current_label = neighbor_label
                                    else:
                                        union_labels(uf_labels, current_label, neighbor_label)

                    if current_label == 0:
                        current_label = label_count
                        label_count += 1

                    labels[z, y, x] = current_label

    # Second Pass: Resolve labels using Union-Find
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if labels[z, y, x] > 0:
                    labels[z, y, x] = find_root(uf_labels, labels[z, y, x])

    # Free memory
    free(uf_labels)

    # Count unique labels
    unique_labels = np.unique(labels)
    num_components = unique_labels.shape[0] - (1 if unique_labels[0] == 0 else 0)

    return np.asarray(labels), num_components
