from dipy.io.image import load_nifti
from dipy.viz import horizon
from thetan import berserker
from thetan.image import img3d_to_ijkv_features
import numpy as np
from thetan.berserker import horizon_support, prepare_data_3d, X_to_many_3D, berserker2_to_3D_plus_extra, centroid_images
from scipy.ndimage import median_filter, gaussian_laplace
# import cc3d


fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
fname = '/home/elef/Data/T1_from_Jasmine/ST18_T1_T1w_MPRAGE_FreeSurfer_0.8iso_20220518080919_8.nii.gz'

data, affine = load_nifti(fname)
print('Data Loaded!')
print(data.shape)

print('Getting TVM!')
voxel_sizes = np.sqrt(np.sum(affine[:3, :3] **2, axis=0))

gi, gj, gk = np.gradient(data, voxel_sizes[0], voxel_sizes[1], voxel_sizes[2])
tvm = np.sqrt(gi ** 2 + gj ** 2 + gk ** 2)

print('World coordinates')
data = np.ascontiguousarray(data, dtype='f8')
X = prepare_data_3d(data, affine, perc=(0,98), mult=300., enh=3.)

print('Normalize and scale TVM')
p_low, p_high = np.percentile(tvm[tvm > 0], [0,98])
tvm_clipped = np.clip(tvm, p_low, p_high)
tvm_norm = (tvm_clipped - p_low) / (p_high - p_low + 1e-12)

tvm_weight = 1
tvm_feat = (tvm_norm.ravel() * tvm_weight).reshape(-1,1)

print('X shape before stacking:', X.shape)
X = np.hstack([X, tvm_feat])
print(f'Feature space with TVM: {X.shape}')

Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)
TVMvol = tvm_norm.reshape(data.shape)
images = horizon_support(affine, (data, 'data'), (W, 'W'), (tvm, 'TVM'), (TVMvol, 'TVMvol'))

horizon(images=images, bg_color=(1,0.5,0.5))

# Max and min values of TVM volume
print('TVM max:', np.max(TVMvol))
print('TVM min:', np.min(TVMvol))

# Gaussian Laplace
Wgrad = gaussian_laplace(W, sigma=1)
images.extend([(Wgrad, affine, 'W_grad')])

# Threshold TVM volume
TVMvol_filtered = TVMvol.copy()
TVMvol_filtered[TVMvol_filtered < 0.4] = 0
images.extend([(TVMvol_filtered, affine, 'TVMvol_filtered')])

print('X shape:', X.shape)
data_labels, centroids, centroids_im, centroids_median_im = berserker2_to_3D_plus_extra(X, W, theta=120.)
images.extend([(data_labels, affine, 'data_labels'), (centroids_im, affine, 'centroids_im'), (centroids_median_im, affine, 'centroids_median_im')])

# Doing clustering only on tvm instead of 3I
X1 = X.copy()
X1[:, 3:] = tvm_feat
tvm_labels, tvm_centroids, tvm_centroids_im, tvm_centroids_median_im = berserker2_to_3D_plus_extra(X1, W, theta=120.)
images.extend([(tvm_labels, affine, 'tvm_labels'), (tvm_centroids_im, affine, 'tvm_centroids_im'), (tvm_centroids_median_im, affine, 'tvm_centroids_median_im')])

# data_labels_cc = cc3d.connected_components(data_labels.astype(np.int32))
# # centroids_im_int = np.round(centroids_im).astype(np.int32)
# centroids_im_cc = cc3d.connected_components(centroids_im)

# # centroids_median_im_int = np.round(centroids_median_im).astype(np.int32)
# centroids_median_im_cc = cc3d.connected_components(centroids_median_im)

# images.extend([(data_labels_cc, affine, 'data_labels_cc'), (centroids_im_cc, affine, 'centroids_im_cc'), (centroids_median_im_cc, affine, 'centroids_median_im_cc')])
# print('data_labels connected unique values length:', len(np.unique(data_labels_cc)))
# print('centroids_im connected unique values length:', len(np.unique(centroids_im_cc)))
# print('centroids_median_im connected unique values length:', len(np.unique(centroids_median_im_cc)))

# Remove TVMvol_filtered from W
W_filtered = W.copy()
W_filtered[TVMvol_filtered > 0] = 0
images.extend([(W_filtered, affine, 'W_filtered')])

# Doing clustering on W_filtered
W_filtered_labels, W_filtered_centroids, W_filtered_centroids_im, W_filtered_centroids_median_im = berserker2_to_3D_plus_extra(X, W_filtered, theta=110.)
images.extend([(W_filtered_labels, affine, 'W_filtered_labels'), (W_filtered_centroids_im, affine, 'W_filtered_centroids_im'), (W_filtered_centroids_median_im, affine, 'W_filtered_centroids_median_im')])

horizon(images=images, bg_color=(1,0.5,0.5))
