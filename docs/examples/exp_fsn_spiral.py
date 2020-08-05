
from time import time
import numpy as np
import pandas as pd
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from robotikos.fsn import fsn_training_with_fgn_shuffles
from robotikos.metric import IdentityFeature

fname = r'C:\Users\elef\Devel\robotikos\data\spirals_with_labels.txt'

data_frame = pd.read_csv(fname, sep="\t", header=None)

df_np = data_frame.to_numpy()
data = df_np[1:, 1:].astype('f8')[:, 1:3]
labels = df_np[1:, 1:].astype('f8')[:, 7]

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

thresholds = [0.05]

x_train_part = data.astype('f4')

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

t = 1

