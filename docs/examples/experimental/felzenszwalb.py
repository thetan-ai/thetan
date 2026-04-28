# import numpy as np
# from scipy.ndimage import gaussian_filter


# class UnionFind:
#     def __init__(self, n):
#         self.parent = np.arange(n)
#         self.size = np.ones(n, dtype=np.int32)
#         self.int_diff = np.zeros(n, dtype=np.float64)

#     def find(self, x):
#         while self.parent[x] != x:
#             self.parent[x] = self.parent[self.parent[x]]
#             x = self.parent[x]
#         return x

#     def union(self, a, b, w):
#         ra, rb = self.find(a), self.find(b)
#         if ra == rb:
#             return False

#         if self.size[ra] < self.size[rb]:
#             ra, rb = rb, ra

#         self.parent[rb] = ra
#         self.size[ra] += self.size[rb]
#         self.int_diff[ra] = w
#         return True


# def _voxel_id(z, y, x, H, W):
#     return z * H * W + y * W + x


# def _get_offsets(connectivity=6):
#     offsets = []

#     for dz in [-1, 0, 1]:
#         for dy in [-1, 0, 1]:
#             for dx in [-1, 0, 1]:
#                 if dz == dy == dx == 0:
#                     continue

#                 dist = abs(dz) + abs(dy) + abs(dx)

#                 if connectivity == 6 and dist == 1:
#                     offsets.append((dz, dy, dx))
#                 elif connectivity == 18 and dist <= 2:
#                     offsets.append((dz, dy, dx))
#                 elif connectivity == 26:
#                     offsets.append((dz, dy, dx))

#     # keep only one direction to avoid duplicate edges
#     offsets = [(dz, dy, dx) for dz, dy, dx in offsets if (dz, dy, dx) > (0, 0, 0)]
#     return offsets


# def felzenszwalb_3d(volume, scale=500, sigma=0.8, min_size=20, connectivity=6):
#     """
#     Felzenszwalb-Huttenlocher segmentation for 3D data.

#     Parameters
#     ----------
#     volume : ndarray
#         Shape (D, H, W) for grayscale volume,
#         or (D, H, W, C) for multichannel volume.
#     scale : float
#         Larger values produce larger segments.
#     sigma : float
#         Gaussian smoothing strength.
#     min_size : int
#         Minimum component size.
#     connectivity : int
#         One of {6, 18, 26}.

#     Returns
#     -------
#     labels : ndarray
#         Integer label volume of shape (D, H, W).
#     """

#     vol = np.asarray(volume, dtype=np.float64)

#     if vol.ndim == 3:
#         if sigma > 0:
#             vol = gaussian_filter(vol, sigma=sigma)
#         features = vol[..., None]

#     elif vol.ndim == 4:
#         if sigma > 0:
#             smoothed = np.empty_like(vol)
#             for c in range(vol.shape[-1]):
#                 smoothed[..., c] = gaussian_filter(vol[..., c], sigma=sigma)
#             vol = smoothed
#         features = vol

#     else:
#         raise ValueError("volume must have shape (D,H,W) or (D,H,W,C)")

#     D, H, W, C = features.shape
#     n_voxels = D * H * W

#     edges = []
#     offsets = _get_offsets(connectivity)

#     for z in range(D):
#         for y in range(H):
#             for x in range(W):
#                 p = _voxel_id(z, y, x, H, W)

#                 for dz, dy, dx in offsets:
#                     zz, yy, xx = z + dz, y + dy, x + dx

#                     if 0 <= zz < D and 0 <= yy < H and 0 <= xx < W:
#                         q = _voxel_id(zz, yy, xx, H, W)

#                         diff = features[z, y, x] - features[zz, yy, xx]
#                         w = np.linalg.norm(diff)

#                         edges.append((w, p, q))

#     edges.sort(key=lambda e: e[0])

#     uf = UnionFind(n_voxels)

#     def threshold(component):
#         root = uf.find(component)
#         return uf.int_diff[root] + scale / uf.size[root]

#     # Main Felzenszwalb merge step
#     for w, a, b in edges:
#         ra, rb = uf.find(a), uf.find(b)

#         if ra != rb:
#             if w <= threshold(ra) and w <= threshold(rb):
#                 uf.union(ra, rb, w)

#     # Merge small components
#     for w, a, b in edges:
#         ra, rb = uf.find(a), uf.find(b)

#         if ra != rb:
#             if uf.size[ra] < min_size or uf.size[rb] < min_size:
#                 uf.union(ra, rb, w)

#     # Convert components to consecutive labels
#     labels = np.zeros(n_voxels, dtype=np.int32)
#     root_to_label = {}
#     current_label = 0

#     for i in range(n_voxels):
#         r = uf.find(i)
#         if r not in root_to_label:
#             root_to_label[r] = current_label
#             current_label += 1
#         labels[i] = root_to_label[r]

#     return labels.reshape(D, H, W)


import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter


@njit
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit
def union(parent, size, internal, a, b, w):
    ra = find(parent, a)
    rb = find(parent, b)

    if ra == rb:
        return

    if size[ra] < size[rb]:
        ra, rb = rb, ra

    parent[rb] = ra
    size[ra] += size[rb]
    internal[ra] = w


@njit
def build_edges_6(volume):
    D, H, W = volume.shape
    n_edges = (D - 1) * H * W + D * (H - 1) * W + D * H * (W - 1)

    weights = np.empty(n_edges, dtype=np.float64)
    a_nodes = np.empty(n_edges, dtype=np.int64)
    b_nodes = np.empty(n_edges, dtype=np.int64)

    e = 0

    for z in range(D):
        for y in range(H):
            for x in range(W):
                p = z * H * W + y * W + x

                if z + 1 < D:
                    q = (z + 1) * H * W + y * W + x
                    weights[e] = abs(volume[z, y, x] - volume[z + 1, y, x])
                    a_nodes[e] = p
                    b_nodes[e] = q
                    e += 1

                if y + 1 < H:
                    q = z * H * W + (y + 1) * W + x
                    weights[e] = abs(volume[z, y, x] - volume[z, y + 1, x])
                    a_nodes[e] = p
                    b_nodes[e] = q
                    e += 1

                if x + 1 < W:
                    q = z * H * W + y * W + (x + 1)
                    weights[e] = abs(volume[z, y, x] - volume[z, y, x + 1])
                    a_nodes[e] = p
                    b_nodes[e] = q
                    e += 1

    return weights, a_nodes, b_nodes


@njit
def segment_sorted_edges(weights, a_nodes, b_nodes, order, n_voxels, scale, min_size):
    parent = np.arange(n_voxels)
    size = np.ones(n_voxels, dtype=np.int64)
    internal = np.zeros(n_voxels, dtype=np.float64)

    # Main Felzenszwalb step
    for idx in order:
        w = weights[idx]
        a = a_nodes[idx]
        b = b_nodes[idx]

        ra = find(parent, a)
        rb = find(parent, b)

        if ra != rb:
            threshold_a = internal[ra] + scale / size[ra]
            threshold_b = internal[rb] + scale / size[rb]

            if w <= threshold_a and w <= threshold_b:
                union(parent, size, internal, ra, rb, w)

    # Merge small components
    for idx in order:
        w = weights[idx]
        a = a_nodes[idx]
        b = b_nodes[idx]

        ra = find(parent, a)
        rb = find(parent, b)

        if ra != rb:
            if size[ra] < min_size or size[rb] < min_size:
                union(parent, size, internal, ra, rb, w)

    return parent


@njit
def relabel(parent):
    n = parent.shape[0]
    labels = np.empty(n, dtype=np.int64)
    root_to_label = -np.ones(n, dtype=np.int64)

    current = 0

    for i in range(n):
        r = find(parent, i)

        if root_to_label[r] == -1:
            root_to_label[r] = current
            current += 1

        labels[i] = root_to_label[r]

    return labels


def felzenszwalb_3d_numba(volume, scale=500, sigma=0.8, min_size=50):
    """
    Fast 3D Felzenszwalb-Huttenlocher segmentation using Numba.

    Parameters
    ----------
    volume : ndarray, shape (D, H, W)
        3D grayscale volume.
    scale : float
        Larger values produce larger regions.
    sigma : float
        Gaussian smoothing.
    min_size : int
        Minimum component size.

    Returns
    -------
    labels : ndarray, shape (D, H, W)
    """

    volume = np.asarray(volume, dtype=np.float64)

    if sigma > 0:
        volume = gaussian_filter(volume, sigma=sigma)

    D, H, W = volume.shape
    n_voxels = D * H * W

    weights, a_nodes, b_nodes = build_edges_6(volume)

    order = np.argsort(weights)

    parent = segment_sorted_edges(
        weights,
        a_nodes,
        b_nodes,
        order,
        n_voxels,
        float(scale),
        int(min_size)
    )

    labels = relabel(parent)

    return labels.reshape(D, H, W)




if __name__ == "__main__":



    from dipy.io.image import load_nifti, save_nifti


    fname = '/home/elef/Data/T1_from_Jasmine/ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz'

    data, affine = load_nifti(fname)

    data = np.ascontiguousarray(data, dtype='f8')

    from thetan.berserker import prepare_data_3d, X_to_many_3D, centroid_images
    from thetan.core import centroids_from_labels

    perc_min, perc_max = 0, 98

    X = prepare_data_3d(data, affine, perc=(perc_min, perc_max), mult=300., enh=3.)
    Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

    labels = felzenszwalb_3d_numba(W, scale=200, sigma=1., min_size=100)

    centroids = centroids_from_labels(labels, W)
    centroids_im = centroid_images(labels, centroids)

    save_nifti('felzenszwalb_labels.nii.gz', labels.astype('int32'), affine)
    save_nifti('felzenszwalb_centroids_im.nii.gz', centroids_im.astype('f8'), affine)
