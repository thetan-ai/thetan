import numpy as np
from dipy.io.image import load_nifti

from thetan.berserker import (
    X_to_many_3D,
    berserker2_to_3D_plus_extra,
    centroids_from_labels_3d,
    prepare_data_3d,
)

fname = r'/home/prajwal/Downloads/t1_data/ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz'

data, affine = load_nifti(fname)

X = prepare_data_3d(data, affine, perc=(0, 98), mult=300., enh=3.)
Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)
W = np.ascontiguousarray(W, dtype='f8')

theta = 60
data_labels_init, centroids_init, centroids_im_init, centroids_median_im_init = berserker2_to_3D_plus_extra(X, W, theta)

data_labels_init = np.ascontiguousarray(data_labels_init, dtype=np.int64)

assert W.ndim == 3
assert data_labels_init.ndim == 3
assert W.shape == data_labels_init.shape

# centroids = centroids_from_labels_3d(W, data_labels_init)

centroids = centroids_from_labels_3d(W, data_labels_init, affine)

print("centroids shape:", centroids.shape)
print("centroids_init shape:", centroids_init.shape)

diff = centroids - centroids_init

print("max abs diff overall:", np.max(np.abs(diff)))
print("max abs diff per column:", np.max(np.abs(diff), axis=0))

bad = np.flatnonzero(
    ~np.isclose(centroids, centroids_init, rtol=1e-5, atol=1e-8).all(axis=1)
)

print("number of bad labels:", len(bad))
print("first bad labels:", bad[:20])

for lab in bad[:5]:
    print("label:", lab)
    print("computed:", centroids[lab])
    print("init:    ", centroids_init[lab])
    print()

assert centroids.shape == centroids_init.shape
assert np.allclose(centroids, centroids_init, atol=1e-5)