import numpy as np
from robotikos.clustering import FastGroupingNode, ClusterMapCentroid, ClusterCentroid
from robotikos.metric import IdentityFeature, Feature
from robotikos.metric import AveragePointwiseEuclideanMetric, EuclideanMetric, SumPointwiseEuclideanMetric


def simple_example_3d():

    data = [0.5 * np.random.rand(4, 3),
            10 + 0.5*np.random.rand(4, 3),
            - 20 + 0.5*np.random.rand(4, 3)]

    # data = np.random.rand(4, 3)

    class PointFeature(Feature):

        def __init__(self):
            super(PointFeature, self).__init__(is_order_invariant=False)

        def infer_shape(self, datum):
            return (4, 3)

        def extract(self, datum):
            return datum

    #point_feature = PointFeature()
    point_feature = IdentityFeature()
    metric = EuclideanMetric(feature=point_feature)
    fgn = FastGroupingNode(thresholds=[10, 5, 0.5], metric=metric)
    tree = fgn.cluster(data)

    for i in [1, 2, 3]:
        print('Level', i)
        clusters = tree.get_clusters(i)
        print(clusters)



if __name__ == '__main__':

    simple_example_3d()