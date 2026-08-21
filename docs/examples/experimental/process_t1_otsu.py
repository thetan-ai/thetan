from time import time

import matplotlib.pyplot as plt
from dipy.io.image import load_nifti

from thetan import scf_2d
from thetan.berserker import enkefalos
from thetan.otsu import local_otsu_2d

fname = r"C:\Users\ipraj\Desktop\brain_seg\ST24_T1w_MPRAGE_FreeSurfer_0.8iso_10_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_10.nii.gz"

data, affine, vxs = load_nifti(fname, return_voxsize=True)
print(data.shape)

print('Starting Thetan Berserker 2 for Brain Segmentation ...')
t = time()

images = enkefalos(data, affine, theta=50., bk=80., diff=30, max_diff=80)

W_mask = images[-9][0]
print('W_mask:', images[-9][2])
W_mask_6 = images[-3][0]
print('W_mask_6:', images[-3][2])
z = 140
print('Slice number',z)
sl = W_mask[:, :, z].astype(float)
print(sl.shape)
t0 = time()

W_mask_thresholded, W_thresholded_map = local_otsu_2d(sl, valid_mask=data[:, :, z] > 0, lower_threshold=3.0, upper_threshold=7.0)

t1 = time()
C = scf_2d(W_mask_thresholded, n_theta=256, sigma=0.25, tau=30.0, dt=2.0)
print(f"SCF done in {time() - t1:.1f}s")

bk = 0.052
C_mask = C > bk

intensity = data[:, :, z]
brain = intensity * C_mask

fig, ax = plt.subplots(1, 6, figsize=(14, 5.5))
ax[0].imshow(intensity.T, cmap="gray", origin="lower")
ax[0].set_title("Original Slice")
ax[1].imshow((W_mask_6[:,:,z].T), cmap = 'gray', origin='lower')
ax[1].set_title('W_mask original thresholded 6')
ax[2].imshow((W_mask_thresholded.T), cmap = 'gray', origin='lower')
ax[2].set_title('Spatial 3-to-7 + bounded Otsu')
ax[3].imshow((C.T) ** 0.4, cmap="magma", origin="lower")  # gamma for visibility
ax[3].set_title("SCF completion field")
ax[4].imshow((C_mask.T), cmap="gray", origin="lower")
ax[4].set_title("Thresholded")
ax[5].imshow(brain.T, cmap="gray", origin="lower")
ax[5].set_title("Masked brain")

for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()

print('Total duration ', time() - t)
print()
