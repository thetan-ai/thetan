import numpy as np

from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (AveragePointwiseEuclideanMetric,
                              SumPointwiseEuclideanMetric,
                              NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

from time import time
from fury import actor, window, ui
from fury.colormap import distinguishable_colormap

fast_hist = True
if fast_hist:
    from fast_histogram import histogram1d
else:
    from numpy import histogram1d


def fgn_image_2d_rgb(data_orig, patch_radius=1, bin_size=5, thresholds=[2.],
                     subimage_width=None, lab_color=False, histo=True,
                     strides=None):
    shuffling = True

    p98 = np.percentile(data_orig, 98)
    data_orig = data_orig.astype('f8')

    # from skimage.color import rgb2lab
    # data_orig = rgb2lab(data_orig)
    print('patch_size', patch_radius * 2 + 1)

    data = np.interp(data_orig, (0, p98), (0, 1.))

    I, J, _ = data.shape

    if subimage_width is not None:
        mI = I // 2
        mJ = J // 2
        w = subimage_width
        data = data[mI - w: mI + w, mJ - w: mJ + w]

    print('data.shape', data.shape)

    # Add padding
    data_r = np.pad(data[..., 0], pad_width=patch_radius, mode='reflect')
    data_g = np.pad(data[..., 1], pad_width=patch_radius, mode='reflect')
    data_b = np.pad(data[..., 2], pad_width=patch_radius, mode='reflect')

    data = np.zeros(data_r.shape + (3,))
    data[..., 0] = data_r
    data[..., 1] = data_g
    data[..., 2] = data_b

    print('data.shape after padding', data.shape)

    print('>> Interpolated and padded')

    class DataSequence2D:
        def __init__(self, data, indices, histo=True):

            self.indices = indices
            self.histo = histo
            self.data = data

        def __getitem__(self, i):

            x, y = self.indices[i]
            patch = self.data[x - patch_radius: x + patch_radius + 1,
                              y - patch_radius: y + patch_radius + 1]

            p1 = patch[:, :, 0]
            p2 = patch[:, :, 1]
            p3 = patch[:, :, 2]

            lp = len(p1)

            if self.histo:
                b1 = histogram1d(p1.ravel(), bins=bin_size, range=(0, 1))
                p1 = b1 / lp
                b2 = histogram1d(p2.ravel(), bins=bin_size, range=(0, 1))
                p2 = b2 / lp
                b3 = histogram1d(p3.ravel(), bins=bin_size, range=(0, 1))
                p3 = b3 / lp

                expand = np.empty((len(p1), 3))
                expand[:, 0] = p1
                expand[:, 1] = p2
                expand[:, 2] = p3
                return np.ascontiguousarray(expand)

            else:
                # compare directly color values
                expand = np.empty((len(p1.ravel()), 3))
                expand[:, 0] = p1.ravel()
                expand[:, 1] = p2.ravel()
                expand[:, 2] = p3.ravel()
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

    if strides is not None:
        internal_indices = internal_indices[:strides]

    internal_indices = np.ascontiguousarray(internal_indices)

    print('>> Shuffled indices')

    ds = DataSequence2D(data, internal_indices, histo=histo)

    print('>> Size of DataSequence2D', len(ds))

    # metric = SumPointwiseEuclideanMetric(feature=IdentityFeature())
    # metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())

    fgn = FastGroupingNode(thresholds=thresholds, metric=metric)

    print('Number of features', len(ds))
    print('Feature size', ds[0].shape)
    print('Thresholds', thresholds)

    t1 = time()
    print('Started clustering')
    fgn_tree = fgn.cluster(ds)

    print('Finished clustering')
    t2 = time()
    print('Duration %.3f' % (t2 - t1,))
    clusters = fgn_tree.get_clusters(1)
    print('Number of clusters %d' % (len(clusters),))

    no_clusters = len(clusters)
    # cluster_colors = np.random.rand(no_clusters, 3)
    # cluster_colors = np.arange(1, no_clusters + 1)
    cluster_colors = distinguishable_colormap(nb_colors=no_clusters)

    final = np.zeros(data.shape)

    for cluster_id, cluster in enumerate(clusters):
        for idx_ in cluster.indices:
            final[tuple(internal_indices[idx_])] = cluster_colors[cluster_id]

    print('Removed extra padding')
    final = final[patch_radius: -patch_radius, patch_radius: -patch_radius]
    data = data[patch_radius: -patch_radius, patch_radius: -patch_radius]

    return final, data


#@profile
def images_2d(interactive=True, patch_radius=1, bin_size=5, thresholds=[2.],
              subimage_width=None, lab_color=False, histo=True, strides=None):

    from imageio import imread
    from os.path import join as pjoin

    dname = '/home/elef/Data/hubmap'
    fname = pjoin(dname, 'original.jpg')
    # fname = pjoin(dname, '188130_web.jpg')
    # fname = pjoin(dname, 'apptop1-10151.jpg')
    # fname = pjoin(dname, 'autumn-landscape-photography-3-900x599.jpg')
    # fname = pjoin(dname, 'forrest.jpg')
    # fname = pjoin(dname, 'Reuben_Wu_Phase_One_XT_1.jpg')
    # fname = pjoin(dname, 'tree_desert.jpg')

    data_orig = imread(fname)

    final, data = fgn_image_2d_rgb(data_orig, patch_radius=patch_radius,
                                   bin_size=bin_size, thresholds=thresholds,
                                   subimage_width=subimage_width,
                                   histo=histo, strides=strides)

    print(final.shape, data.shape, data_orig.shape)

    if interactive:

        scene = window.Scene()
        scene.add(actor.texture((255 * data).astype('uint8')))

        final_actor = actor.texture((255 * final).astype('uint8'))
        final_actor.SetPosition(data.shape[0] + 50, 0, 0)

        final_actor2 = actor.texture(
                (255 * (0.5 * final + 0.5 * data)).astype('uint8'))
        final_actor2.SetPosition(2 * data.shape[0] + 2 * 50, 0, 0)

        scene.add(final_actor)
        scene.add(final_actor2)
        window.show(scene)


if __name__ == '__main__':

    # images_2d(True, patch_radius=5, bin_size=5, thresholds=[20.])
    # images_2d(True, patch_radius=5, bin_size=5, thresholds=[40.])
    # images_2d(True, patch_radius=3, bin_size=5, thresholds=[15.])
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[1.],
    #           subimage_width=200, histo=True)
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[1.],
    #           subimage_width=None, histo=True)
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[.2],
    #           subimage_width=200, histo=False)
    images_2d(True, patch_radius=5, bin_size=5, thresholds=[.1],
              subimage_width=None, histo=False, strides=100000)
