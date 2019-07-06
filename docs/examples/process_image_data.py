

import numpy as np
from robotikos.clustering import QuickBundlesX
from robotikos.metric import IdentityFeature, Feature
from robotikos.metric import AveragePointwiseEuclideanMetric, EuclideanMetric, SumPointwiseEuclideanMetric
from robotikos.metric import CrossCorrelationMetric
from dipy.io.image import load_nifti, save_nifti
from dipy.viz.app import horizon
from robotikos.metric import Metric
from time import time



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


def images_3d(run_id):

    dname = 'C:/Users/elef/Data/ds031/sub-01/ses-106/anat/'
    # dname = '/home/elef/Data/ds031/sub-01/ses-106/anat/'
    fname = dname + 'sub-01_ses-106_T1w.nii.gz'
    # fname = dname + 'sub-01_ses-106_T2w.nii.gz'

    interactive = True
    sim = False
    magnitude = False
    punch = False
    histo = True
    shuffling = True
    small_data = False
    sobel_support = False
    patch_radius = 5

    # thresholds = [20.]  # for standard 10 GOOOD # 20 Goood 3 clusters
    # thresholds = [250.]  # for sobel
    thresholds = [2.] # [3.4] # for histogram

    # if sim:
    #    thresholds = [0.2]

    print(fname)

    data_orig, affine = load_nifti(fname)
    
    if sim:
        data_orig = np.zeros(data_orig.shape, dtype='f8')
        data_orig[50-10 : 50 + 10, 50 - 10 : 50 + 10, 50 - 10 : 50 + 10] = 255

    from pylab import figure, imshow, show, plot, title

    if interactive:
        figure()
        title('T1 orig')
        imshow(data_orig[..., 50], cmap='gray')
        #show()

    if magnitude:
        from scipy import ndimage
        data_orig = ndimage.gaussian_gradient_magnitude(data_orig, sigma=5)
        save_nifti('magnitude.nii.gz', data_orig, affine)

    p98 = np.percentile(data_orig, 98)
    data_orig = data_orig.astype('f8')
        
    data = np.interp(data_orig, (0, p98), (0, 1.))
    
    if sobel_support:
        from scipy import ndimage
        sx = ndimage.sobel(data, axis=0, mode='constant')
        sy = ndimage.sobel(data, axis=1, mode='constant')
        sz = ndimage.sobel(data, axis=2, mode='constant')
        data = np.sqrt(sx**2 + sy**2 + sz**2)
        # data = np.interp(data, (0, p98), (0, 1.))
    
    save_nifti('interpolated.nii.gz', data, affine)


    if sim:
        data = np.interp(data_orig, (0, 255), (0, 1.))

    # save_nifti('discrete.nii.gz', data, affine)

    if interactive:
        figure()
        title('T1 0-1')
        imshow(data[..., 50], cmap='gray')
        # show()

    I, J, K = data.shape

    

    class DataSequence:
        def __init__(self, data, indices, histo=False):

            self.indices = indices
            self.histo = histo
            self.data = data

        def __getitem__(self, i):

            if i % 10000 == 0:
                print(i)

            x, y, z = self.indices[i]
            patch = self.data[x - patch_radius: x + patch_radius + 1,
                              y - patch_radius: y + patch_radius + 1,
                              z - patch_radius: z + patch_radius + 1]
            
            # p = patch.ravel()[None, :].astype('f4')
            # p = np.ascontiguousarray(p)
            
            p = patch.ravel()
            if self.histo:
                b = np.histogram(p, bins=10, range=(0, 1))
                p = b[0].astype('f8') / len(p)

            expand = np.zeros((len(p), 3))
            expand[:, 0] = p
            expand[:, 1] = p
            expand[:, 2] = p
            return np.ascontiguousarray(expand)

        def __len__(self):
            return len(self.indices)

    
    if small_data:
        data = data[:, :, 50 - patch_radius: 50 + patch_radius + 1]

    internal_indices = []
    X, Y, Z = data.shape

    for idx in np.ndindex(*data.shape):

        x, y, z = idx
        if x >= patch_radius and x < X - patch_radius:
            if y >= patch_radius and y < Y - patch_radius:
                if z >= patch_radius and z < Z - patch_radius:
                    internal_indices.append(idx)

    internal_indices = np.array(internal_indices)
    
    if shuffling:
        np.random.shuffle(internal_indices)

    internal_indices = np.ascontiguousarray(internal_indices)

    # metric = CrossCorrelationMetric(feature=IdentityFeature())

    from dipy.segment.featurespeed import IdentityFeature

    feature = IdentityFeature()

    feature.is_order_invariant = True

    print(feature.is_order_invariant)

    from dipy.segment.metric import EuclideanMetric

    metric = EuclideanMetric(feature=feature) # This is actually the sum Eucl.

    ds = DataSequence(data, internal_indices, histo=histo)

    print('Size of ds', len(ds))

    # ds_list = [ds[cnt] for cnt in range(len(indices))]

    # Below 0.3 group together
    

    from dipy.segment.clustering import QuickBundles

    # qbx = QuickBundlesX(thresholds, metric=metric)
    qb = QuickBundles(threshold=thresholds[0], metric=metric)

    # from pdb import set_trace
    # set_trace()


    t1 = time()
    print('Started clustering')
    # tree = qbx.cluster(ds)
    clusters = qb.cluster(ds)
    print('Finished clustering')
    t2 = time()
    print('Duration %.3f' % (t2 - t1,))
    # clusters = tree.get_clusters(1)
    print('Number of clusters %d' % (len(clusters),))

    no_clusters = len(clusters)
    # cluster_colors = np.random.rand(no_clusters, 3)

    cluster_colors = np.arange(1, no_clusters + 1)

    final = np.zeros(data.shape) - 1

    for cluster_id, cluster in enumerate(clusters):
        for idx_ in cluster.indices:
            final[tuple(internal_indices[idx_])] = cluster_colors[cluster_id]
    
    if not small_data:
        save_nifti('final_' + str(run_id) + '_' + str(thresholds[0]) + '_' + str(sobel_support) + '_' + str(patch_radius) + '_' + str(histo) + '.nii.gz', final, affine)

    if small_data:
        figure()
        title('Grouped')
        imshow(final[..., patch_radius], cmap='jet')
        show()
        """
        horizon(tractograms=[], images=[(final, np.eye(4))],
                cluster=False, cluster_thr=15,
                random_colors=False,
                length_lt=10, length_gt=100,
                clusters_lt=5000, clusters_gt=10000,
                world_coords=True, interactive=True)
        """
    # from pdb import set_trace
    # set_trace()




    if punch:
        csizes = []
        for c in clusters:
            csizes.append(len(c))
        
        asort = np.argsort(csizes)
        punch_value = asort[-1]
        print('punch value ', punch_value)
        new_data = data.copy()
        new_data[final == punch_value + 1] = 0

        from pdb import set_trace
        set_trace()
        save_nifti('new_data_punched.nii.gz', new_data, affine)
        from dipy.segment.mask import median_otsu

        from scipy import ndimage
        # structure = ndimage.generate_binary_structure(3, 1)
        binary = new_data.copy()
        binary[binary > 0] = 1
        binary[binary < 0] = 0

        structure2 = np.ones((5, 5, 5))

        # label, num_features = ndimage.label(binary, structure=structure)
        # print(num_features)
        opening = ndimage.binary_opening(binary, structure2)

        # masked_data, mask = median_otsu(new_data)
        # save_nifti('masked_data.nii.gz', masked_data, affine)
        # save_nifti('generated_mask.nii.gz', mask.astype('uint8'), affine)
        save_nifti('opening.nii.gz', opening.astype('uint16'), affine)



if __name__ == '__main__':

    # vectors_3d()
    repeat_times = 2
    for i in range(1, repeat_times):
        images_3d(i)
