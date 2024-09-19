import os
import numpy as np
from glob import glob
from thetan.berserker import berserker2_3D_rt
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import horizon


dname = '/tmp/HCP_labels/'
search_fname = dname + '*_T1w.nii.gz'
verbose = False
theta_value = 150. # 120.

res = sorted(glob(search_fname))[:10]

info = np.zeros((10, 5))

for i, fname in enumerate(res):

    subj_id = os.path.basename(fname).split('.nii.gz')[0].split('_')[0]

    print()
    print('# Participant id ', subj_id)

    fname_t1 = dname + subj_id + '_T1w.nii.gz'
    fname_seg = dname + subj_id + '_seg.nii.gz'

    t1, t1_aff = load_nifti(fname_t1)
    seg, seg_aff = load_nifti(fname_seg)

    data_labels, centroids, centroids_im, W, rt = berserker2_3D_rt(t1, t1_aff, theta_value=theta_value, perc=(0, 98), mult=300., enh=3.)

    print('Number of labels ', centroids.shape[0])

    images = [(t1, t1_aff, 't1'), (seg, seg_aff, 'seg'), (data_labels, t1_aff, 'dl'), (centroids_im, t1_aff, 'cim'), (W, t1_aff, 'W')]

    if verbose:
        horizon(images=images)

    wm_mask = (seg == 41) | (seg == 2)

    overlap = data_labels[wm_mask == True]

    unq, cnts = np.unique(overlap.ravel(), return_counts=True)

    print('>>>> ', unq, cnts)

    ratio = cnts.max() / wm_mask.sum()
    print('>##> ', ratio)

    info[i, 0] = rt
    info[i, 1] = centroids.shape[0]
    info[i, 2] = ratio

t = 1