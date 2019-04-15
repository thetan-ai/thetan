import numpy as np
from robotikos.clustering import QuickBundlesX
from robotikos.metric import IdentityFeature, Feature
from robotikos.metric import AveragePointwiseEuclideanMetric, EuclideanMetric, SumPointwiseEuclideanMetric
from robotikos.metric import CrossCorrelationMetric
from dipy.io.image import load_nifti, save_nifti
from dipy.viz.app import horizon
from robotikos.metric import Metric
from time import time


#    horizon(tractograms=[], images=[data],
#            cluster=False, cluster_thr=15,
#            random_colors=False,
#            length_lt=10, length_gt=100,
#            clusters_lt=5000, clusters_gt=10000,
#            world_coords=True, interactive=True)


def vectors_3d():

    data = [np.random.rand(1, 3),
            np.random.rand(1, 3),
            np.random.rand(1, 3)]

    class PointFeature(Feature):

        def __init__(self):
            super(PointFeature, self).__init__(is_order_invariant=False)

        def infer_shape(self, streamline):
            return (1, 3)

        def extract(self, streamline):
            return streamline

    point_feature = PointFeature()
    point_feature = IdentityFeature()
    metric = EuclideanMetric(feature=point_feature)
    qbx = QuickBundlesX(thresholds=[10], metric=metric)
    tree = qbx.cluster(data)
    clusters = tree.get_clusters(1)

    print(clusters)

    sampling = 'uniform'

    if sampling == 'normal':
        my_data_1 = 10 * np.random.normal(size=(1000, 3))
        my_data_2 = 10 * np.random.normal(size=(1000, 3)) + np.array([250, 0, 0])
        my_data_3 = 10 * np.random.normal(size=(1000, 3)) - np.array([250, 0, 0])

    step = 0

    if sampling == 'uniform':
        my_data_1 = 200 * np.random.rand(1000, 3)
        my_data_2 = 200 * np.random.rand(1000, 3) + np.array([200 + step, 0, 0])
        my_data_3 = 200 * np.random.rand(1000, 3) - np.array([200 + step, 0, 0])

    my_data = np.concatenate((my_data_1, my_data_2, my_data_3))
    my_data = np.ascontiguousarray(my_data)

    print(my_data.shape)

    class DataSequence:
        def __init__(self, data):

            self.data = data

        def __getitem__(self, i):
            return self.data[i][None, :]

        def __len__(self):
            return self.data.shape[0]

    dataseq = DataSequence(my_data)

    vector_feature = IdentityFeature()
    metric = EuclideanMetric(feature=vector_feature)

    thresholds = [250]
    # thresholds = [125]

    qbx = QuickBundlesX(thresholds=thresholds, metric=metric)
    tree = qbx.cluster(dataseq)
    clusters = tree.get_clusters(1)
    print(len(clusters))

    from fury import actor, window, ui

    scene = window.Renderer()
    spheres = actor.sphere(centers=my_data, colors=np.ones((len(my_data), 3)))
    scene.add(spheres)
    window.show(scene)

    new_colors = np.zeros((len(my_data), 3))
    no_clusters = len(clusters)
    cluster_colors = np.random.rand(no_clusters, 3)
    for c, cluster in enumerate(clusters):
        for ind in cluster.indices:
            new_colors[ind] = cluster_colors[c]

    scene = window.Renderer()
    spheres = actor.sphere(centers=my_data, colors=new_colors)
    scene.add(spheres)
    window.show(scene)


    switch_data_rep = False

    if switch_data_rep:

        dataexp = my_data.copy()
        dataexp.shape = (my_data.shape[0], 1, my_data.shape[1])

        ''' alternative way
        tmp = my_data[:, None, :]
        tmp.flags
        tmp = np.ascontiguousarray(tmp)
        '''

        indices = np.random.randint(0, dataexp.shape[0], size=dataexp.shape[0])

        tree = qbx.cluster(dataexp,
                           ordering=indices)
        clusters = tree.get_clusters(1)
        print(len(clusters))


def ncc(v1, v2, z=False, absolute=True, ord=2):

    vec1 = v1.ravel()
    if z:
        vec1 = vec1 - np.mean(vec1)
    nvec1 = np.linalg.norm(vec1, ord=ord)

    vec2 = v2.ravel()
    if z:
        vec2 = vec2 - np.mean(vec2)
    nvec2 = np.linalg.norm(vec2, ord=ord)

    if absolute:
        return np.abs(np.dot(vec1, vec2) / (nvec1 * nvec2))
    return np.dot(vec1, vec2) / (nvec1 * nvec2)


def images_3d():

    dname = 'C:/Users/elef/Data/ds031/sub-01/ses-106/anat/'
    fname = dname + 'sub-01_ses-106_T1w.nii.gz'
    # fname = dname + 'sub-01_ses-106_T2w.nii.gz'

    interactive = True

    print(fname)

    data_orig, affine = load_nifti(fname)

    from pylab import figure, imshow, show, plot, title

    if interactive:
        figure()
        title('T1 orig')
        imshow(data_orig[..., 50], cmap='gray')
        #show()

    p98 = np.percentile(data_orig, 98)
    data_orig = data_orig.astype('f8')
    data = np.interp(data_orig, (0, p98), (0, 1.))

    # save_nifti('discrete.nii.gz', data, affine)

    if interactive:
        figure()
        title('T1 0-1')
        imshow(data[..., 50], cmap='gray')
        # show()

    I, J, K = data.shape

    patch_radius = 3

    class DataSequence:
        def __init__(self, indices):

            self.indices = indices

        def __getitem__(self, i):

            x, y, z = self.indices[i]
            patch = data[x - patch_radius: x + patch_radius + 1,
                         y - patch_radius: y + patch_radius + 1,
                         z - patch_radius: z + patch_radius + 1]
            p = patch.ravel()[None, :].astype('f4')
            p = np.ascontiguousarray(p)
            return p

        def __len__(self):
            # return patch_radius * 2 + 1
            return len(self.indices)

    small_data = True

    if small_data:
        data = data[:, :, 50 - patch_radius:50 + patch_radius + 1]

    internal_indices = []

    X, Y, Z = data.shape

    for idx in np.ndindex(*data.shape):

        x, y, z = idx
        if x >= patch_radius and x < X - patch_radius:
            if y >= patch_radius and y < Y - patch_radius:
                if z >= patch_radius and z < Z - patch_radius:
                    internal_indices.append(idx)

    internal_indices = np.array(internal_indices)

    shuffling = True

    if shuffling:
        shuffled_indices = np.random.randint(0, len(internal_indices),
                                             size=internal_indices.shape[0])
        internal_indices = internal_indices[shuffled_indices]

    internal_indices = np.ascontiguousarray(internal_indices)

    # metric = CrossCorrelationMetric(feature=IdentityFeature())

    metric = EuclideanMetric(feature=IdentityFeature())

    ds = DataSequence(internal_indices)

    print('Size of ds', len(ds))

    # ds_list = [ds[cnt] for cnt in range(len(indices))]

    # Below 0.3 group together
    # thresholds = [0.1]
    thresholds = [4.]

    qbx = QuickBundlesX(thresholds, metric=metric)

    t1 = time()
    print('Started clustering')
    tree = qbx.cluster(ds)
    print('Finished clustering')
    t2 = time()
    print('Duration %.3f' % (t2 - t1,))
    clusters = tree.get_clusters(1)
    print('Number of clusters %d' % (len(clusters),))

    no_clusters = len(clusters)
    # cluster_colors = np.random.rand(no_clusters, 3)
    cluster_colors = np.arange(0, no_clusters)

    final = np.zeros(data.shape)

    for cluster_id, cluster in enumerate(clusters):
        for idx_ in cluster.indices:
            final[tuple(internal_indices[idx_])] = cluster_colors[cluster_id]

    if not small_data:
        save_nifti('final3.nii.gz', final, affine)

    if small_data:
        figure()
        title('Grouped')
        imshow(final[..., patch_radius])
        show()

    1/0


if __name__ == '__main__':

    vectors_3d()
    # images_3d()
