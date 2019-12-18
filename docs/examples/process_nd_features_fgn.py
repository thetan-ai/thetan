import numpy as np
from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (AveragePointwiseEuclideanMetric,
                              SumPointwiseEuclideanMetric,
                              NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)
from time import time


def simple_example_3d(interactive=True):

    data = [0.5 * np.random.rand(400, 3),
            2 + 0.5*np.random.rand(400, 3),
            - 4 + 0.5*np.random.rand(400, 3)]

    data = np.random.rand(4*10**2, 40, 300)
    data[2*10**2:] += 2

    # If data is 2D (e.g. array of points) then do
    # data = data[:, None, :]

    if interactive:
        from fury import actor, window

        scene = window.Scene()
        for i in range(len(data)):
            scene.add(actor.point(data[i], colors=(1, 0, 0)))
        window.show(scene)

    class DataSequence2D:
        def __init__(self, data, indices):

            self.indices = indices
            self.data = data

        def __getitem__(self, i):

            return self.data[self.indices[i]]

        def __len__(self):
            return len(self.indices)

    internal_indices = np.arange(len(data))

    np.random.seed(42)
    np.random.shuffle(internal_indices)

    ds = DataSequence2D(data, internal_indices)

    # metric = SumPointwiseEuclideanMetric(feature=IdentityFeature())
    # metric = AveragePointwiseEuclideanMetric(feature=IdentityFeature())
    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    # metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())

    print('Example distance (float)',
          metric.dist(ds[0].astype('f4'), ds[1].astype('f4')))

    # NSSD
    print('Expected NSSD distance (double)',
          np.sum((ds[0] - ds[1]) ** 2) / np.prod(ds[0].shape))

    # NSAD
    print('Expected NSAD distance (double)',
          np.sum(np.abs(ds[0] - ds[1])) / np.prod(ds[0].shape))

    # from pdb import set_trace
    # set_trace()

    # thresholds = [2, 1, 0.5]
    # thresholds = [0.5, 0.4, 0.3, 0.2, 0.19, 0.18, 0.17, 0.165, 0.16, 0.15, 0.1]
    thresholds = [2, 1, 0.5, 0.35, 0.1]

    fgn = FastGroupingNode(thresholds=thresholds, metric=metric)

    t1 = time()
    print('Number of points', len(ds))
    print('Feature size', ds[0].shape)
    print('Thresholds', thresholds)
    print('Started clustering FGN')

    fgn_tree = fgn.cluster(ds)

    print('Finished clustering')
    t2 = time()
    print('Duration %.3f' % (t2 - t1,))

    for i in range(len(thresholds)):
        print('Level', i + 1)
        clusters = fgn_tree.get_clusters(i + 1)
        if interactive:
            print('#clusters', clusters)
        print(len(clusters))


if __name__ == '__main__':

    simple_example_3d(interactive=False)