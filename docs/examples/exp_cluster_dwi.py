import numpy as np
from dipy.io.image import load_nifti, save_nifti
from robotikos.fgm import fgn_shuffles, fgn_ma
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)


# fname = r"C:\\Users\\elef\\.dipy\\stanford_hardi\\HARDI150.nii.gz"
# fout = r"C:\\Users\\elef\\.dipy\\stanford_hardi\\HARDI150_out.nii.gz"
fname = r"/home/elef/.dipy/stanford_hardi/HARDI150.nii.gz"

theta = 0.0001 * 0.5 * 0.2 * 0.8
num_shuffles = 10

print('Threshold ', theta)
print('Number of Shuffles ', num_shuffles)

fout = r"/home/elef/.dipy/stanford_hardi/HARDI150_om_" + str(theta) + "_" + str(num_shuffles) + ".nii.gz"
flabels = r"/home/elef/.dipy/stanford_hardi/HARDI150_om_" + str(theta) + "_" + str(num_shuffles) + ".txt"

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



from time import time

t1 = time()
res = fgn_shuffles(X, thresholds=[theta],
                   metric=metric,
                   num_shuffles=num_shuffles, mode='fgn_ma', verbose=True)

t2 = time()
print('Done in ', t2 - t1)

level = sorted(res.keys())[-1]
centroids = res[level]['centroids']
labels = res[level]['labels']
un, cnts = np.unique(labels, return_counts=True)
print('Number of clusters ', len(centroids.keys()))
tmp = 1
X2 = np.zeros(X.shape, dtype='f4')
X_cnt = np.zeros(X.shape[:3], dtype='f4')

for vox_id, l in enumerate(labels):
    X2[vox_id] = centroids[l].copy()
    # we could save here the number of counts
    # here I am directly copying the centroids
    # but perhaps better strategy is to weight 
    # the contributions of signals of that cluster that are
    # more similar to the current signal.

X2 = X2 * xmax
X2 = X2.reshape(data.shape)
save_nifti(fout, X2, affine)
tmp2 = 2
print('File saved at ', fout)

np.savetxt(flabels, labels)
print('Labels saved at ', flabels)
