from time import time
import numpy as np
from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (AveragePointwiseEuclideanMetric,
                              SumPointwiseEuclideanMetric,
                              NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

# data with two clusters
power = 5
data = np.random.rand(4 * 10**power, 2, 3)
data[2 * 10 ** power:] += 4

thresholds = [2.]
interactive = False

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())

fgn = FastGroupingNode(thresholds, metric)


print('Number of data', len(data))
print('Number of features', np.prod(data[0].shape))
print('Feature shape', data[0].shape)

print('Standard FGN version')

t1 = time()
fgn_tree = fgn.cluster(data)
t2 = time()
print('Duration %.3f s' % (t2 - t1,))


for i in range(len(thresholds)):
    print('Level', i + 1)
    clusters = fgn_tree.get_clusters(i + 1)
    if interactive:
        print('# clusters', clusters)
    print(len(clusters))


feature_shape = data[0].shape

fgn_next = fgn.cluster_online(feature_shape)

print('Online FGN version')

t1 = time()
for i in range(len(data)):
    fgn_online_tree, path = fgn_next(data[i], i)
t2 = time()
print('Duration %.3f s' % (t2 - t1,))

fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

for i in range(len(thresholds)):
    print('Level', i + 1)
    clusters = fgn_online_all_levels.get_clusters(i + 1)
    if interactive:
        print('# clusters', clusters)
    print(len(clusters))

