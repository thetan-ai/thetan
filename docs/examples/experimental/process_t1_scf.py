from time import time

import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import load_nifti

from thetan import scf_2d
from thetan.berserker import enkefalos
from thetan.viz import plot_scf

fname = r"C:\Users\ipraj\Desktop\brain_seg\ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz"

data, affine, vxs = load_nifti(fname, return_voxsize=True)
data = np.ascontiguousarray(data, dtype="f8")

print('Starting Thetan Berserker 2 for Brain Segmentation ...')
t = time()
images = enkefalos(data, affine, theta=50., bk=80., diff=30, max_diff=80)

W_cv_mask = images[-5][0]
print(images[-5][2])

C_2d_to_3d = np.zeros_like(W_cv_mask)

for sli in range(W_cv_mask.shape[2]):
    C_2d = scf_2d(W_cv_mask[..., sli], n_theta=64, sigma=0.5, tau=12, dt=1, tensor_sigma=0.1)
    C_2d_to_3d[..., sli] = C_2d

print(C_2d_to_3d.shape, C_2d_to_3d.max(), C_2d_to_3d.min())
images += [(C_2d_to_3d, affine, 'SCF 2D to 3D')]

print('Total duration ', time() - t)
print()

# For plotting:
slice_numbers = [140, 160]
fig = plot_scf(data, W_cv_mask, C_2d_to_3d, slice_numbers=slice_numbers, threshold_value=0.052)
plt.show()

# 3D version:
# t1 = time()
# C_3d = scf_3d(W_cv_blur_mask, n_theta=16, n_phi=8, sigma=0.25,tau=12, dt=1)
# print(f'SCF3d done in {time() - t1:.1f}s')

# images += horizon_support(aff, (C_3d, "SCF_3D"))
# skyline(images=images)
