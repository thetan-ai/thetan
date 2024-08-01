import numpy as np
from thetan.berserker import berserker2_recovery
from dipy.io.image import load_nifti, save_nifti

# fname = 'C:/Users/elef/Downloads/Marcus1.nii.gz'
# fname2 = 'C:/Users/elef/Downloads/Marcus1_sr.nii.gz'

fname = 'C:/Users/elef/.dipy/tissue_data/t1_brain.nii.gz'
# fname = 'C:/Users/elef/.dipy/tissue_data/t1_brain_denoised.nii.gz'

# fname2 = 'C:/Users/elef/Downloads/t1_brain_r2.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

rec, W, medrec = berserker2_recovery(data, affine, theta_value=100, perc=(0, 98), mult=300, enh=3, reps=15, medroids=True)

# save_nifti(fname2, rec, affine)

rec2, W, medrec2 = berserker2_recovery(data, affine, theta_value=100, perc=(0, 98), mult=300, enh=3, reps=15, medroids=True)


rec12 = 0.5 * (rec + rec2)
medrec12 = 0.5 * (medrec + medrec2)

diff = np.abs(W - rec12)
diff2 = np.abs(W - medrec12)
diff3 = np.abs(rec12 - medrec12)

from dipy.viz import horizon
horizon(images=[(data, affine, 'original'),
                (W, affine, 'W'), 
                (rec, affine, 'rec'), 
                (medrec, affine, 'medrec'), 
                (rec2, affine, 'rec2'), 
                (medrec2, affine, 'medrec2'),
                (rec12, affine, 'rec12'),
                (medrec12, affine, 'medrec12'), 
                (diff, affine, 'W - rec12'),
                (diff2, affine, 'W - medrec12'),
                (diff3, affine, 'rec12 - medrec12')
                ])



t = 1