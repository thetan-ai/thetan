import numpy as np
from clobber import UnionFind, compute_cluster_stats, find_neighbor_pairs, remap_labels
from dipy.io.image import load_nifti
from dipy.viz import skyline
from scipy.ndimage import median_filter

from thetan.berserker import (
    X_to_many_3D,
    berserker2_to_3D_plus_extra,
    centroid_images,
    prepare_data_3d,
)
from thetan.patchify import patchify

path = r'/home/prajwal/Downloads/IXI002-Guys-0828-T1.nii.gz'

theta = 80
threshold = 10
max_intensity_threshold = 30
patch_size = (3, 3, 3)

data, affine = load_nifti(path)

X = prepare_data_3d(data, affine, perc=(0,98), mult=300, enh=3)
Ivol, Jvol, Kvol = X_to_many_3D(X, data.shape)
W = np.ascontiguousarray(W, dtype='f8')

data_labels_orig, centroids_orig, centroids_im_orig, centroids_median_im_orig = berserker2_to_3D_plus_extra(X, W, theta)

label_patches = patchify(data_labels_orig, patch_size)
data_patches = patchify(data, patch_size)

label_patches = np.ascontiguousarray(label_patches, dtype=np.int64)
data_patches = np.ascontiguousarray(data_patches, dtype=np.float64)

n_labels = int(data_labels_orig.max()) + 1
labels_i64 = np.ascontiguousarray(data_labels_orig, dtype=np.int64)
data_f64 = np.ascontiguousarray(data, dtype=np.float64)

cluster_intensities_arr, cluster_sizes_arr = compute_cluster_stats(labels_i64, data_f64, n_labels)

unique_labels = np.unique(data_labels_orig)
centroid_intensities_arr = np.zeros(n_labels, dtype=np.float64)
for i, lab in enumerate(unique_labels):
    centroid_intensities_arr[lab] = centroids_orig[i, 3]

n0, n1, n2 = label_patches.shape[:3]
neighbor_pairs, skipped = find_neighbor_pairs(label_patches, data_patches)

print(
    f"Total patches: {n0 * n1 * n2}, "
    f"skipped: {skipped}, "
    f"Processed: {n0 * n1 * n2 - skipped}"
)

merge_candidates = []
for a, b in sorted(neighbor_pairs):
    diff = abs(cluster_intensities_arr[a] - cluster_intensities_arr[b])
    if diff < threshold:
        merge_candidates.append((a,b,cluster_intensities_arr[a], cluster_intensities_arr[b], diff))

print(f"Total neighbor pairs: {len(neighbor_pairs)}")
print(f"Merge candidates (diff < {threshold}): {len(merge_candidates)}\n")

print(f"{'Cluster A':>10} {'Cluster B':>10} {'Mean A':>10} {'Mean B':>10} {'Diff':>10}")

for a, b, mean_a, mean_b, diff in merge_candidates:
    print(f"{a:10d} {b:10d} {mean_a:10.2f} {mean_b:10.2f} {diff:10.2f}")

merge_candidates.sort(key=lambda x: min(centroid_intensities_arr[x[0]], centroid_intensities_arr[x[1]]))

uf = UnionFind(n_labels, np.ascontiguousarray(cluster_sizes_arr, dtype=np.int64), np.ascontiguousarray(centroid_intensities_arr, dtype=np.float64))

rejected = 0
for a, b, _, _, _ in merge_candidates:
    if not uf.union(a, b, centroid_intensities_arr[a], centroid_intensities_arr[b], max_intensity_threshold):
        rejected += 1

print(
    f"Merges rejected (would exceed max_intensity_threshold="
    f"{max_intensity_threshold}): {rejected}"
)

label_map = np.arange(n_labels, dtype=np.int64)
for lab in range(n_labels):
    label_map[lab] = uf.find(lab)

unique_roots = np.unique(label_map)
compact = np.zeros(n_labels, dtype=np.int64)
for new_id, root in enumerate(unique_roots):
    compact[root] = new_id
full_map = np.array([compact[label_map[i]] for i in range(n_labels)], dtype=np.int64)

data_labels_merged = np.ascontiguousarray(np.copy(labels_i64), dtype=np.int64)
remap_labels(data_labels_merged, full_map)

n_before = len(np.unique(data_labels_orig))
n_after = len(np.unique(data_labels_merged))
print(f"\nClusters before merging: {n_before}")
print(f"Clusters after merging:  {n_after}")
print(f"Clusters reduced by:     {n_before - n_after}")

centroids_im_orig = centroid_images(data_labels_merged, centroids_orig)
centroids_im_median_merged = median_filter(centroids_im_orig, size=3)

print(f"data labels merged, {n_after} clusters remaining")
print(f"data labels merged max, {np.max(data_labels_merged)}")
print(f"data labels merged min, {np.min(data_labels_merged)}")

skyline(
    images=[
        (data, affine, "data"),
        (W, affine, "W"),
        (data_labels_orig, affine, "data_labels_orig"),
        (data_labels_merged, affine, "data_labels_merged"),
        (centroids_im_orig, affine, "centroids_im_orig"),
        (centroids_median_im_orig, affine, "centroids_median_im_orig"),
        (centroids_im_merged, affine, "centroids_im_merged"),
        (centroids_im_median_merged, affine, "centroids_im_median_merged"),
    ]
)