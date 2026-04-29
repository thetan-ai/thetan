import numpy as np
import metaimageio
from thetan.berserker import X_to_many_3D, berserker2_to_3D_plus_extra, prepare_data_3d, horizon_support
from dipy.viz import skyline


draw = "C:/Users/Eleftherios/Data/S-BIAD2106/SELMA3D2026_training_annotated/contiguous_structures/raw/"
dgt = "C:/Users/Eleftherios/Data/S-BIAD2106/SELMA3D2026_training_annotated/contiguous_structures/gt/"

# fname = draw + "peripheral_nerve_patchvolume_227.mha"
# flabel = dgt + "peripheral_nerve_patchvolume_227.mha"

fname = draw + "blood_vessel_patchvolume_010.mha"
flabel = dgt + "blood_vessel_patchvolume_010.mha"

[data, meta] = metaimageio.read(fname)
[label, meta2] = metaimageio.read(flabel)



show = True
theta = 120.

affine = np.eye(4)
images=[(data, affine, 'raw'), (label, affine, 'gt')]


data = np.ascontiguousarray(data, dtype='f8')

print('Preparing data')

X = prepare_data_3d(data, affine, perc=(0, 98), mult=300., enh=3.)
Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

print('Running Thetan Berserker 2 for Initial Segmentation ...')
data_labels, centroids, centroids_im, centroids_median_im = berserker2_to_3D_plus_extra(X, W, theta)
images += horizon_support(affine, (W, 'W'), (data_labels, 'dl'), (centroids_im, 'cim'), (centroids_median_im, 'cmim'))

print('Centroids Shape:', centroids.shape)




if show:
    skyline(images=images)




