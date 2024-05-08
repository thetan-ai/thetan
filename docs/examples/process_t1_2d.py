from time import time
from dipy.io.image import load_nifti
from dipy.viz import horizon
from thetan.berserker import berserker2_brain_seg_2d

fname = 'C:/Work/Repos/gens/IXI002-Guys-0828-T1.nii.gz'



data, affine, vxs = load_nifti(fname, return_voxsize=True)

print('Starting Thetan Berserker 2 for Brain Segmentation ...')
t = time()
images = berserker2_brain_seg_2d(data, affine, return_all=True, orig_est=100, m_est=3, theta_value=120./2, percb=5)
print('Total duration ', time() - t)
print()

horizon(images=images)


t = 2

print('Done')

