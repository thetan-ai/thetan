import numpy as np
from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (AveragePointwiseEuclideanMetric,
                              SumPointwiseEuclideanMetric,
                              NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

# data with two clusters
data = np.random.rand(4 * 10**1, 40, 300)
data[2 * 10 ** 1:] += 4

thresholds = [2.]
interactive = True

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

fgn = FastGroupingNode(thresholds, metric)

fgn_tree = fgn.cluster(data)

print('Standard version')

for i in range(len(thresholds)):
    print('Level', i + 1)
    clusters = fgn_tree.get_clusters(i + 1)
    if interactive:
        print('# clusters', clusters)
    print(len(clusters))


feature_shape = data[0].shape

fgn_next = fgn.cluster_online(feature_shape)

print('Online version')

for i in range(len(data)):

    fgn_online_tree, path = fgn_next(data[i], i)

    # online_all_levels = fgn_online_tree.get_tree_cluster_map()

    # online_all_levels.get_clusters(1)

fgn_online_all_levels = fgn_online_tree.get_tree_cluster_map()

for i in range(len(thresholds)):
    print('Level', i + 1)
    clusters = fgn_online_all_levels.get_clusters(i + 1)
    if interactive:
        print('# clusters', clusters)
    print(len(clusters))