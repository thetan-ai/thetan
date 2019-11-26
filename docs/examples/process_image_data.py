

import numpy as np
from robotikos.clustering import QuickBundlesX, ClusterMapCentroid, ClusterCentroid
from robotikos.metric import IdentityFeature, Feature
from robotikos.metric import AveragePointwiseEuclideanMetric, EuclideanMetric, SumPointwiseEuclideanMetric
from robotikos.metric import CrossCorrelationMetric
from dipy.io.image import load_nifti, save_nifti
from dipy.viz.app import horizon
from robotikos.metric import Metric
from time import time
from fury import actor, window, ui
from itertools import cycle, islice


def sklearn_data():
    from sklearn import cluster, datasets, mixture
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                        'quantile': .2, 'n_clusters': 2,
                        'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    all_data = {}

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset
        from pdb import set_trace
        # set_trace()
        my_data = np.zeros((X.shape[0], 3))
        my_data[:, :2] = X.copy()
        print(my_data.shape)

        all_data[i_dataset] = my_data

    return all_data


def simple_example_3d():
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


def vectors_3d(run=0, mode='skdata', sampling='uniform', skdata_id=0, thresholds=[250.], sph_radius=0.1,
               fstealth=None, init_viz=False, shuffling=True, chain_maker=True):
    """ skdata or ecubes """



    if mode == 'ecubes':
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

    if mode == 'skdata':

        all_data = sklearn_data()
        my_data = all_data[skdata_id]
        # TODO: Attension only using a small dataset
        my_data = my_data[:200]

    if mode == 'packing':

        from pix_coord import generate_circles
        my_data = generate_circles(inner_radius=40, outer_radius=15) # Try 60.
        my_data = np.asarray(my_data[1])
        tmp = np.zeros((len(my_data), 3))
        tmp[:, :2] = my_data
        my_data = np.ascontiguousarray(tmp)

        #from pdb import set_trace
        #set_trace()

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

    qbx = QuickBundlesX(thresholds=thresholds, metric=metric)

    internal_indices = np.arange(my_data.shape[0])

    if shuffling:
        np.random.seed(run)
        np.random.shuffle(internal_indices)

    internal_indices = np.ascontiguousarray(internal_indices)
    print(internal_indices)

    tree = qbx.cluster(dataseq, ordering=internal_indices)
    clusters = tree.get_clusters(1)
    print(len(clusters))

    print('Clusters before chaining')
    print(clusters)

    """ legacy needs more thinking
    if chain_maker:
        # qbx_clusters = qbx.cluster(sample_streamlines, ordering=indices)

        print(' Chaining is starting ...')
        cluster_map = ClusterMapCentroid()

        current_cluster_id = 0
        visited = []
        visited_labels = []
        not_visited = list(range(len(clusters)))

        new_clusters = {}
        new_clusters[current_cluster_id] = 0
        # from pdb import set_trace
        # set_trace()
        global CNT
        CNT = 0

        def recursive_chaining(current_cluster_id, clusters, visited, not_visited, expanding_cluster=None, cluster_map=None):

            # print('Recursive Chaining')
            # print(cluster_map)
            global CNT
            print(CNT)
            print(expanding_cluster)
            CNT += 1

            if len(not_visited) == 0:
                # print('All visited')
                return

            if expanding_cluster is None:
                expanding_cluster = ClusterCentroid(centroid=clusters[0].centroid)
                expanding_cluster.indices.extend(clusters[current_cluster_id].indices)
                # cluster_map.add_cluster(expanding_cluster)
            else:
                expanding_cluster.indices.extend(clusters[current_cluster_id].indices)

            visited.append(current_cluster_id)
            not_visited.remove(current_cluster_id)
            current_cluster = clusters[current_cluster_id]
            dists = []

            for c in not_visited:

                cluster = clusters[c]
                d = metric.dist(current_cluster.centroid,
                                cluster.centroid)

                if d <= thresholds[0]:
                    # print('Close')
                    dists.append(c)
                else:
                    # print('Not close')
                    pass

            if len(dists) == 0:
                # cluster_map.add_cluster(expanding_cluster)

                if len(not_visited) > 0:
                    next_cluster_id = not_visited[0]
                    recursive_chaining(next_cluster_id, clusters, visited,
                                       not_visited, None, cluster_map)
                else:
                    return

            else:
                try:
                    next_index = np.argmin(dists)
                    next_cluster_id = not_visited[next_index]
                    recursive_chaining(next_cluster_id, clusters, visited,
                                       not_visited, expanding_cluster, cluster_map)

                except ValueError as e:
                    print(e)
                    from pdb import set_trace
                    set_trace()



        recursive_chaining(current_cluster_id, clusters, visited, not_visited, None, cluster_map)

        cluster_map.refdata = my_data
        clusters = cluster_map

    """
    #print('Final cluster_map')
    #print(cluster_map)

    # from pdb import set_trace
    # set_trace()

    from fury import actor, window, ui

    init_viz = True
    if init_viz:
        scene = window.Renderer()
        spheres = actor.sphere(centers=my_data, colors=np.ones((len(my_data), 3)), radii=sph_radius)
        scene.add(spheres)
        window.show(scene)

    new_colors = np.zeros((len(my_data), 3))
    no_clusters = len(clusters)
    cluster_colors = np.random.rand(no_clusters, 3)
    for c, cluster in enumerate(clusters):
        for ind in cluster.indices:
            new_colors[ind] = cluster_colors[c]

    scene = window.Renderer()
    spheres = actor.sphere(centers=my_data, colors=new_colors, radii=sph_radius)
    scene.add(spheres)
    if fstealth is None:
        window.show(scene)
    else:
        window.snapshot(scene, fstealth)

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

def timeseries():

    import pandas as pd
    import matplotlib.pyplot as plt

    dname = '/home/elef/Data/heart_rate/'
    dataset = pd.read_csv(dname + "data.csv") #Read data from CSV datafile
    plt.title("Heart Rate Signal") #The title of our plot
    plt.plot(dataset.hart) #Draw the plot object
    plt.show() #Display the plot



def images_3d(run_id, interactive=True, sim=False,
              magnitude=False,
              punch=False,
              histo=True,
              shuffling=True,
              small_data=False,
              sobel_support=False,
              patch_radius=5,
              thresholds=[2.],
              padding=True):

    # ----------------------------------------------------------------------------------
    # TODO show overlaps and play with threshold wich for histogram is currently at 2.0
    # TODO show results with T2.
    # TODO try with rounded patch not rectangular
    # ----------------------------------------------------------------------------------

    # dname = 'C:/Users/elef/Data/ds031/sub-01/ses-106/anat/'
    dname = '/home/elef/Data/ds031/sub-01/ses-106/anat/'
    fname = dname + 'sub-01_ses-106_T1w.nii.gz'
    # fname = dname + 'sub-01_ses-106_T2w.nii.gz'

    # interactive = True
    # sim = False
    # magnitude = False
    # punch = False
    # histo = True
    # shuffling = True
    # small_data = False
    # sobel_support = False
    # patch_radius = 5

    # thresholds = [20.]  # for standard 10 GOOOD # 20 Goood 3 clusters
    # thresholds = [250.]  # for sobel
    # thresholds = [2.] # [3.4] # for histogram

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

    if padding:

        data_orig = np.pad(data_orig, pad_width=patch_radius, mode='constant', constant_values=0)


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

            # if i % 10000 == 0:
            #    print(i)

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

    # unpad
    final = final[patch_radius:-patch_radius, patch_radius:-patch_radius, patch_radius:-patch_radius]


    if not small_data:
        save_nifti('new_final_' + str(run_id) + '_thr_' + str(thresholds[0]) + '_sob_' + str(sobel_support) + '_pr_' + str(patch_radius) + '_histo_' + str(histo) + '.nii.gz', final, affine)

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



def images_2d(interactive=True, patch_radius=1, bin_size=10, thresholds=[2.]):

    from imageio import imread
    from os.path import join as pjoin

    dname = '/home/elef/Data/hubmap'
    fname = pjoin(dname, 'original.jpg')
    shuffling = True

    data_orig = imread(fname)

    p98 = np.percentile(data_orig, 98)
    data_orig = data_orig.astype('f8')

    data = np.interp(data_orig, (0, p98), (0, 1.))

    print('>> Interpolated')

    from fury import actor, window

    scene = window.Scene()
    scene.add(actor.texture((255 * data).astype('uint8')))

    if interactive:
        window.show(scene)

    I, J, K = data.shape

    class DataSequence2D:
        def __init__(self, data, indices, histo=False):

            self.indices = indices
            self.histo = histo
            self.data = data

        def __getitem__(self, i):

            x, y = self.indices[i]
            patch = self.data[x - patch_radius: x + patch_radius + 1,
                              y - patch_radius: y + patch_radius + 1]
            1/0
            p = patch.ravel()

            if self.histo:
                b = np.histogram(p, bins=bin_size, range=(0, 1))
                p = b[0].astype('f8') / len(p)

            expand = np.zeros((len(p), 3))
            expand[:, 0] = p
            expand[:, 1] = p
            return np.ascontiguousarray(expand)

        def __len__(self):
            return len(self.indices)

    internal_indices = []
    X, Y, _ = data.shape

    for idx in np.ndindex(*data.shape[:2]):

        x, y = idx
        if x >= patch_radius and x < X - patch_radius:
            if y >= patch_radius and y < Y - patch_radius:
                internal_indices.append(idx)

    internal_indices = np.array(internal_indices)

    print('>> Internal indices')

    if shuffling:
        np.random.shuffle(internal_indices)

    internal_indices = np.ascontiguousarray(internal_indices)

    print('>> Shuffled indices')

    from dipy.segment.featurespeed import IdentityFeature
    feature = IdentityFeature()

    feature.is_order_invariant = True

    print(feature.is_order_invariant)

    from dipy.segment.metric import EuclideanMetric

    metric = EuclideanMetric(feature=feature) # This is actually the sum Eucl.

    ds = DataSequence2D(data, internal_indices, histo=True)

    print('>> Size of DataSequence2D', len(ds))

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

    # unpad
    final = final[patch_radius:-patch_radius, patch_radius:-patch_radius]

    final_actor = actor.texture(final)
    final_actor.SetPosition(0, data.shape[0] + 500, 0)
    scene.add(final_actor)
    window.show(scene)

    1/0


if __name__ == '__main__':

    images_2d()

    1/0

    thr = 25
    skdata_id = 0
    fstealth = None
    chain_maker = False
    vectors_3d(run=0, mode='packing', skdata_id=0, thresholds=[thr], sph_radius=1, init_viz=False,
               fstealth=fstealth, shuffling=True, chain_maker=chain_maker)

    1/0


    for i in range(1):
        if fstealth is not None:
            fname = 'vec3d_run' + str(i) + '_id_' + str(skdata_id) + '_' + str(thr) + '.png'
        else:
            fname = None
        print(fname)
        vectors_3d(run=i, mode='skdata', skdata_id=skdata_id, thresholds=[thr], sph_radius=0.05, init_viz=False,
                   fstealth=fname, shuffling=True, chain_maker=chain_maker)

    # segment 3d images
    """
    # for pr in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # for pr in [2, 3, 4, 6, 7, 9]:
    for pr in [1, 5, 8, 10]:
        print('Patch radius ', pr)
        images_3d(1, histo=True, patch_radius=pr, thresholds=[2.0])
    """



