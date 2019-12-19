import numpy as np
from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (AveragePointwiseEuclideanMetric,
                              SumPointwiseEuclideanMetric,
                              NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)


data = np.random.rand(4*10**2, 40, 300)
data[2*10**2:] += 2

thresholds = [2., 1., 0.5, 0.1]
interactive = False

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

fgn = FastGroupingNode(thresholds, metric)

fgn_tree = fgn.cluster(data)

for i in range(len(thresholds)):
    print('Level', i + 1)
    clusters = fgn_tree.get_clusters(i + 1)
    if interactive:
        print('#clusters', clusters)
    print(len(clusters))


fgn_stepper = fgn.cluster_online(data)
