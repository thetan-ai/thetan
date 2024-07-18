from thetan.berserker import berserker2_recovery
from dipy.io.image import load_nifti, save_nifti

# fname = 'C:/Users/elef/Downloads/Marcus1.nii.gz'
# fname2 = 'C:/Users/elef/Downloads/Marcus1_sr.nii.gz'

fname = 'C:/Users/elef/.dipy/tissue_data/t1_brain.nii.gz'
fname2 = 'C:/Users/elef/Downloads/t1_brain_r2.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

rec, W = berserker2_recovery(data, affine, theta_value=100, perc=(0, 98), mult=300, enh=3, reps=2)

save_nifti(fname2, rec, affine)

rec2, W = berserker2_recovery(data, affine, theta_value=100, perc=(0, 98), mult=300, enh=3, reps=2)

save_nifti(fname2, rec2, affine)

from dipy.viz import horizon

horizon(images=[(data, affine, 'original'), (W, affine, 'W'), (rec, affine, 'recovery'), (rec2, affine, 'recovery2')])

t = 1