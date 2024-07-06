from thetan.berserker import berserker2_recovery
from dipy.io.image import load_nifti, save_nifti

fname = 'C:/Users/elef/Downloads/Marcus1.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

rec = berserker2_recovery(data, affine, theta_value=120, perc=(0, 98), mult=300, enh=3, reps=2)

from dipy.viz import horizon

horizon(images=[(data, affine, 'original'), (rec, affine, 'recovery')])

t = 1