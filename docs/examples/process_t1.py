import platform

from os.path import join as pjoin
from time import time
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import horizon
from thetan.berserker import berserker2_brain_seg, berserker2_brain_seg_mini, exponentiation_normalization, plot_exp_norm
from sklearn.cluster import KMeans


# T1 dataset
# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
# fname = '/home/elef/Downloads/T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz'
# fname = '/home/elef/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'
# fname = '/home/elef/Data/T1_from_Jasmine/ST18_T1_T1w_MPRAGE_FreeSurfer_0.8iso_20220518080919_8.nii.gz'

# fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'

# fname = 'C:/Users/elef/Data/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/S00/t1_ori.nii.gz'
# fname = 'C:/Users/elef/Data/S00/b0_mean.nii.gz'
# fname = 'C:/Users/elef/Data/T1_from_Jasmine/ST18_T1_T1w_MPRAGE_FreeSurfer_0.8iso_20220518080919_8.nii.gz'

# fname = 'C:/Users/elef/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'

# leopard
fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/For_Jasmine_Mirdama/T1/T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz'
# fname = 'C:/Users/Eleftherios/.dipy/stanford_hardi/t1.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/T1_from_Jasmine/S06_T1w_MPRAGE_FreeSurfer_0.8iso_7_T1w_MPRAGE_FreeSurfer_0.8iso_20191021131954_7.nii.gz'


# if platform.node() == 'Serges-MacBook-Pro.local':
#     folder = '/Users/skoudoro/data/thetan_data/thetan_data'
#     fname = pjoin(folder, 'IXI002-Guys-0828-T1.nii.gz')
    # fname = pjoin(folder, 'IXI012-HH-1211-T1.nii.gz')
    # fname = pjoin(folder, 'T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz')
    # fname = 'C:/Users/Eleftherios/.dipy/stanford_hardi/t1.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

# DWI dataset
# fname = '/home/elef/.dipy/stanford_hardi/dwi_patch2self.nii.gz'
# fname = 'C:/Users/Eleftherios/.dipy/stanford_hardi/HARDI150.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# # Just keep b0 for now
# data = data[..., 0].copy()

# Binfinity
# fname = '/home/elef/Data/distortions/b0_u.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/distortions/b0_u.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# fmask = '/home/elef/Data/distortions/b0_u_thetan2.nii.gz'

# B0 HCP (LR)
# fname = '/home/elef/Data/distortions/103818_3T_DWI_dir96_LR.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/distortions/103818_3T_DWI_dir96_LR.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# fmask = '/home/elef/Data/distortions/103818_3T_DWI_dir96_LR_thetan2.nii.gz'
# fmask = 'C:/Users/Eleftherios/Data/distortions/103818_3T_DWI_dir96_LR_thetan2.nii.gz'
# data = data[..., 0].copy()

# Simulation example
# data, affine, vxs = simulate_volumes(sh=(100, 100, 100), radii=[10, 20, 30, 40], values=[10, 20, 30, 40.], pos=[(50, 50, 50), (50, 50, 50), (50, 50, 50), (50, 50, 50)])
# data, affine, vxs = simulate_volumes(sh=(100, 100, 100), radii=[10, 20, 30, 40], values=[5, 5, 5, 5.], pos=[(50, 50, 50), (50, 50, 50), (50, 50, 50), (50, 50, 50)])

# horizon(images=[(data, np.eye(4), 'orig')])

# plot_exp_norm()

print('Starting Thetan Berserker 2 for Brain Segmentation ...')
t = time()
# images, images2 = berserker2_brain_seg(data, affine, return_all=True, orig_est=100, m_est=3, theta_value=120., percb=5, pg=False, ng=False)

images = berserker2_brain_seg_mini(data, affine, theta_value=120., alpha=2, scale=9, bk=20., bk2=400.)

print('Total duration ', time() - t)
print()

horizon(images=images, bg_color=(1, 0.5, 0.5))
print('Done')

t = 1

