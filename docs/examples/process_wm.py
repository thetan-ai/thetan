import os
from glob import glob
from thetan.berserker import berserker2_3D
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import horizon


dname = '/tmp/HCP_labels/'
search_fname = dname + '*_T1w.nii.gz'

res = sorted(glob(search_fname))[:10]

fname = res[0]

subj_id = os.path.basename(fname).split('.nii.gz')[0].split('_')[0]

print(subj_id)

fname_t1 = dname + subj_id + '_T1w.nii.gz'
fname_seg = dname + subj_id + '_seg.nii.gz'

t1, t1_aff = load_nifti(fname_t1)
seg, seg_aff = load_nifti(fname_seg)

theta_value = 150. # 120.

data_labels, centroids, centroids_im, W = berserker2_3D(t1, t1_aff, theta_value=theta_value, perc=(0, 98), mult=300., enh=3.)

print('Number of labels ', centroids.shape[0])

images = [(t1, t1_aff, 't1'), (seg, seg_aff, 'seg'), (data_labels, t1_aff, 'dl'), (centroids_im, t1_aff, 'cim'), (W, t1_aff, 'W')]

horizon(images=images)


t = 1