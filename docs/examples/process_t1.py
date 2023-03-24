
import numpy as np
from thetan.theta import img3d_to_ijkv_features
from dipy.io.image import load_nifti
from dipy.viz.app import horizon
from thetan.berserker import berserker2


fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'


data, affine = load_nifti(fname)



data =  np.ascontiguousarray(data, dtype='f8')

X = img3d_to_ijkv_features(data)

# Calculating z-scores. It will help with metric spaces

X = (X - np.mean(X, axis=0) / np.std(X, axis=0))

# horizon(images=[(data, affine)])


ind = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=False)
X_shuffled = X[ind].copy()
ind_inv = np.argsort(ind)

centroids, labels = berserker2(X_shuffled, 50, all_out=False)

labels_unsh = labels[ind_inv].copy()

data_labels = labels_unsh.copy()

data_labels.reshape(*data.shape)

horizon(images=[(data, affine)])

horizon(images=[(data_labels, affine)])

t = 1
