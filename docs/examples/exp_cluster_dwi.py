import numpy as np
from dipy.io.image import load_nifti, save_nifti
from robotikos.fgm import fgn_shuffles, fgn_ma
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)


fname = r"C:\\Users\\elef\\.dipy\\stanford_hardi\\HARDI150.nii.gz"
fout = r"C:\\Users\\elef\\.dipy\\stanford_hardi\\HARDI150_out.nii.gz"

data, affine = load_nifti(fname)

X = data.copy()

X = X.reshape((np.prod(X.shape[:3]), 1, X.shape[3]))

xmax = X.max()

X = X / xmax

# X = X[:50000]

mX = np.mean(X, axis=-1)

# from pylab import figure, plot, show

# figure()
# plot(mX.ravel())
# show()

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

theta = 0.0001 * 0.5 * 0.2 * 0.1

from time import time

t1 = time()
res = fgn_shuffles(X, thresholds=[theta],
                   metric=metric,
                   num_shuffles=1, mode='fgn_ma', verbose=True)

t2 = time()
print('Done in ', t2 - t1)

level = sorted(res.keys())[-1]
fcentroids = res[level]['centroids']
flabels = res[level]['labels']
print(len(fcentroids.keys()))
tmp = 1
X2 = np.zeros(X.shape, dtype='f4')
for vox_id, l in enumerate(flabels):
    X2[vox_id] = fcentroids[l].copy()

X2 = X2 * xmax
X2 = X2.reshape(data.shape)
save_nifti(fout, X2, affine)
tmp2 = 2