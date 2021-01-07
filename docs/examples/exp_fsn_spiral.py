
from time import time
import numpy as np
import pandas as pd
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from robotikos.fsn import fsn_training_with_fgn_shuffles
from robotikos.fsn import fsn_test_with_fgn_shuffles
from robotikos.metric import IdentityFeature

fname = r'C:\Users\elef\Devel\robotikos\data\spirals_with_labels.txt'

data_frame = pd.read_csv(fname, sep="\t", header=None)

df_np = data_frame.to_numpy()
data = df_np[1:, :].astype('f8')[:, 1:3]
labels = df_np[1:, :].astype('f8')[:, 8]

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

thresholds = [0.005]

x_train_part = np.ascontiguousarray(data[:, None, :].astype('f4'))

print(x_train_part)

labels[labels == -1] = 0

y_train_part = np.ascontiguousarray(labels).astype('i8')

num_shuffles = 10

level = 1

verbose = True

t1 = time()
centroids, y_train_pred = fsn_training_with_fgn_shuffles(
    x_train_part, y_train_part, thresholds, metric,
    num_shuffles=num_shuffles, mode='fgn_ma',
    verbose=verbose, level=level, num_labels=2)
t2 = time()

print(centroids)

x_test_part = 25 * (np.random.rand(10000, 2) - 0.5)

x = np.linspace(-12.5, 12.5, 100)

# numpy.linspace creates an array of
# 9 linearly placed elements between
# -4 and 4, both inclusive
y = np.linspace(-12.5, 12.5, 100)

# The meshgrid function returns
# two 2-dimensional arrays
x_1, y_1 = np.meshgrid(x, y)

x_1 = x_1.ravel()
y_1 = y_1.ravel()

x_test_part = np.zeros((len(x_1), 2))
x_test_part[:, 0] = x_1
x_test_part[:, 1] = y_1
x_test_part = np.ascontiguousarray(x_test_part)

x_test_part = x_test_part[:, None, :]

t1 = time()
y_test_pred = fsn_test_with_fgn_shuffles(x_test_part, centroids, metric)
t2 = time()

print('Duration FGN shuffles testing %.3f s' % (t2 - t1,))

import pylab as pl

pl.scatter(x_test_part[:, 0, 0], x_test_part[:, 0, 1], c=y_test_pred, marker='s', alpha=0.1)

pl.scatter(x_train_part[:, 0, 0], x_train_part[:, 0, 1], c=y_train_part)


pl.show()

t = 1

