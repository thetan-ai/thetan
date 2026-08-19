from time import time

import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import load_nifti
from dipy.viz import skyline
from scipy.ndimage import gaussian_filter

from thetan import scf_2d, scf_3d
from thetan.berserker import X_to_many_3D, horizon_support, prepare_data_3d
from thetan.patchify import patchify

fname = r"C:\Users\ipraj\Desktop\brain_seg\ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz"

data, affine, vxs = load_nifti(fname, return_voxsize=True)
data = np.ascontiguousarray(data, dtype="f8")

print("Preparing data")
perc_min = 0
perc_max = 98
X = prepare_data_3d(data, affine, perc=(perc_min, perc_max), mult=300.0, enh=3.0)
Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)
aff = affine.copy()

W = np.ascontiguousarray(W, dtype='f8')
images = horizon_support(aff, (W, "W"))

W_min = W.min()
W_max = W.max()

W_patches = patchify(W, (3,3,3), step=1)
W_patches_mean = W_patches.mean(axis=(3,4,5))
W_patches_var = W_patches.var(axis=(3,4,5))
images += horizon_support(
    aff, (W_patches_mean, "W_patches_mean"), (W_patches_var, "W_patches_var")
)

W_patches_cv = W_patches_var / (W_patches_mean + 1e-12)
images += horizon_support(aff, (W_patches_cv, "W_patches_cv"))

W_patches_cv_blur = gaussian_filter(W_patches_cv, sigma=1)
images += horizon_support(aff, (W_patches_cv_blur, "W_patches_cv_blur"))

pad_width = 1
W_patches_cv_blur = np.pad(W_patches_cv_blur, pad_width=1, mode='constant', constant_values=0)

W_cv_blur = W / (W_patches_cv_blur + 1e-12)
images += horizon_support(aff, (W_cv_blur, "W_cv_blur"))
W_cv_blur_mask = W_cv_blur < 6 # THis is the current threshold that i am checking with
W_cv_blur_mask = np.ascontiguousarray(
    W_cv_blur_mask,
    dtype=np.float32,
)
images += horizon_support(aff, (W_cv_blur_mask, "W_cv_blur_mask"))

# 2D version:
z = 140
sl = W_cv_blur_mask[:, :, z].astype(float)
t1 = time()
C = scf_2d(sl, n_theta=256, sigma=0.25, tau=30.0, dt=2.0)
print(f'SCF2d done in {time() - t1:.1f}s')

bk = 0.052
C_mask = C > bk

fig, ax = plt.subplots(1, 3, figsize=(14, 5.5))
ax[0].imshow(data[:, : ,z].T, cmap="gray", origin="lower")
ax[0].set_title("Original Slice")
ax[1].imshow((sl.T), cmap = 'gray', origin='lower')
ax[1].set_title('CV_blur_mask')
ax[2].imshow((C.T) ** 0.4, cmap="magma", origin="lower")
ax[2].set_title("SCF completion field")

for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()


# 3D version:
t1 = time()
C_3d = scf_3d(W_cv_blur_mask, n_theta=16, n_phi=8, sigma=0.25,tau=12, dt=1)
print(f'SCF3d done in {time() - t1:.1f}s')

images += horizon_support(aff, (C_3d, "SCF_3D"))
skyline(images=images)
