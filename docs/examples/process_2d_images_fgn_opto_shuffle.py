import numpy as np

from robotikos.clustering import FastGroupingNode
from robotikos.metric import IdentityFeature
from robotikos.metric import (AveragePointwiseEuclideanMetric,
                              SumPointwiseEuclideanMetric,
                              NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric,
                              NormalizedSumInfDiffMetric)

from time import time
from fury import actor, window, ui
from fury.colormap import distinguishable_colormap
from robotikos.fgm import fgn, fgn_shuffles
from cv2 import imwrite

fast_hist = True
if fast_hist:
    from fast_histogram import histogram1d
else:
    from numpy import histogram1d


def fgn_image_2d_rgb(data_orig, patch_radius=1, bin_size=5, thresholds=[2.],
                     subimage_width=None, lab_color=False, histo=True,
                     n_shuffles = 10, random_selection=None, strides=False,
                     distance='nssd', interpolation='0-98p', only_centers=True):
    #shuffling = True

    data_orig = data_orig.astype('f8')

    # from skimage.color import rgb2lab
    # data_orig = rgb2lab(data_orig)
    print('patch_size', patch_radius * 2 + 1)

    if interpolation == '0-98p':
        p98 = np.percentile(data_orig, 98)
        data = np.interp(data_orig, (0, p98), (0, 1.))
    if interpolation == '0-max':
        data = np.interp(data_orig, (0, data_orig.max()), (0, 1.))
    if interpolation == 'min-max':
        data = np.interp(data_orig, (data_orig.min(), data_orig.max()), (0, 1.))
    if interpolation == '0-255':
        data = np.interp(data_orig, (0, 255), (0, 1.))

    print('>> Interpolated')

    I, J, _ = data.shape

    if subimage_width is not None:
        mI = I // 2
        mJ = J // 2
        w = subimage_width
        data = data[mI - w: mI + w, mJ - w: mJ + w]

    print('data.shape', data.shape)

    if patch_radius > 0:
        data_r = np.pad(data[..., 0], pad_width=patch_radius, mode='reflect')
        data_g = np.pad(data[..., 1], pad_width=patch_radius, mode='reflect')
        data_b = np.pad(data[..., 2], pad_width=patch_radius, mode='reflect')
        print('>> Padding added')

        data = np.zeros(data_r.shape + (3,))
        data[..., 0] = data_r
        data[..., 1] = data_g
        data[..., 2] = data_b

    p_data = []
    for x in range(patch_radius, data.shape[0]-patch_radius):
        for y in range(patch_radius, data.shape[1]-patch_radius):
            patch = data[x - patch_radius: x + patch_radius + 1,
                              y - patch_radius: y + patch_radius + 1]

            if patch_radius > 0:
                p1 = patch[:, :, 0]
                p2 = patch[:, :, 1]
                p3 = patch[:, :, 2]
                lp = len(p1)
            else:

                p1, p2, p3 = patch[0, 0]
                lp = 1

            if histo:
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
                p_data.append(np.ascontiguousarray(expand))

            else:
                if patch_radius > 0:
                    # compare directly color values
                    expand = np.empty((len(p1.ravel()), 3))
                    expand[:, 0] = p1.ravel()
                    expand[:, 1] = p2.ravel()
                    expand[:, 2] = p3.ravel()
                else:
                    expand = np.array([[p1, p2, p3]])
                p_data.append(np.ascontiguousarray(expand))
    
    p_data = np.asarray(p_data)
    print('data.shape', data.shape)
    print('preprocessed data shape', p_data.shape)

    if distance.lower() == 'spe':
        metric = SumPointwiseEuclideanMetric(feature=IdentityFeature())
    elif distance.lower() == 'nssd':
        metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    elif distance.lower() == 'nsad':
        metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())
    elif distance.lower() == 'nsid':
        metric = NormalizedSumInfDiffMetric(feature=IdentityFeature())
    else:
        raise ValueError('Unknown distance')

    print('Thresholds', thresholds)

    t1 = time()
    print('Started clustering')
    collect1 = fgn_shuffles(p_data, thresholds, metric=metric,
                        num_shuffles=n_shuffles, mode='fgn')

    print('Finished clustering')
    t2 = time()
    print('Duration %.3f' % (t2 - t1,))

    t1 = time()
    print('Started clustering')
    collect2 = fgn_shuffles(p_data, thresholds, metric=metric,
                        num_shuffles=n_shuffles, mode='fgn_ma')

    print('Finished clustering')
    t2 = time()
    print('Duration %.3f' % (t2 - t1,))

    clusters = collect1[1]['labels']
    clusters_ma = collect2[1]['labels']
    print(len(clusters))
    print(len(collect1[1]['centroids'].keys()))
    print(len(collect2[1]['centroids'].keys()))

    # TODO: generate results for all clusters not only the first cluster
    #clusters = fgn_tree.get_clusters(1)
    #centroids = fgn_tree.centroids()
    #cluster_list[i] = clusters


    no_clusters = len(clusters)
    no_clusters_ma = len(clusters_ma)
    #print(clusters[0])
    # cluster_colors = np.random.rand(no_clusters, 3)
    # cluster_colors = np.arange(1, no_clusters + 1)
    cluster_colors = distinguishable_colormap(nb_colors=no_clusters)
    cluster_colors_ma = distinguishable_colormap(nb_colors=no_clusters_ma)

    if patch_radius > 0:
        print('Removed extra padding')
        #final = final[patch_radius: -patch_radius, patch_radius: -patch_radius]
        data = data[patch_radius: -patch_radius, patch_radius: -patch_radius]
        #final_ma = final_ma[patch_radius: -patch_radius, patch_radius: -patch_radius]

    final = np.zeros(data.shape)
    final_ma = np.zeros(data.shape)

    #print(cluster_colors[0])
    #print(cluster_colors[1])


    for ind in range(len(clusters)):
        final[ind//data.shape[1], ind%data.shape[1]] = cluster_colors[clusters[ind]]

    for ind in range(len(clusters_ma)):
        final_ma[ind//data.shape[1], ind%data.shape[1]] = cluster_colors_ma[clusters_ma[ind]]


    return final, final_ma, data


#@profile
def images_2d(data_orig, interactive=True, patch_radius=1, bin_size=5,
              thresholds=[2.], n_shuffles=10,
              subimage_width=None, lab_color=False, histo=True,
              random_selection=1000000, strides=False, distance='nsad',
              interpolation='98p', only_centers=True):

    final, final_ma, data = fgn_image_2d_rgb(data_orig, patch_radius=patch_radius,
                                   bin_size=bin_size, thresholds=thresholds,
                                   subimage_width=subimage_width,
                                   histo=histo, n_shuffles=n_shuffles,
                                   random_selection=random_selection,
                                   strides=strides, distance=distance,
                                   interpolation=interpolation,
                                   only_centers=only_centers)

    print(final.shape, final_ma.shape, data.shape, data_orig.shape)

    imwrite('final.jpg', (255 * final).astype('uint8'))
    imwrite('final_ma.jpg', (255 * final_ma).astype('uint8'))

    if interactive:

        scene = window.Scene()
        data_actor = actor.texture((255 * data).astype('uint8'),
                                   interp=False)
        scene.add(data_actor)

        final_actor = actor.texture((255 * final).astype('uint8'),
                                    interp=False)
        final_actor.SetPosition(data.shape[0] + 50, 0, 0)

        final_actor2 = actor.texture(
                (255 * (0.5 * final + 0.5 * data)).astype('uint8'),
                interp=False)

        final_actor2.SetPosition(2 * data.shape[0] + 2 * 50, 0, 0)

        final_ma_actor = actor.texture((255 * final_ma).astype('uint8'),
                                    interp=False)
        
        final_ma_actor.SetPosition(3 * data.shape[0] + 3 * 50, 0, 0)

        final_ma_actor2 = actor.texture(
                (255 * (0.5 * final_ma + 0.5 * data)).astype('uint8'),
                interp=False)
        
        final_ma_actor2.SetPosition(4 * data.shape[0] + 4 * 50, 0, 0)

        scene.add(final_actor)
        scene.add(final_actor2)
        scene.add(final_ma_actor)
        scene.add(final_ma_actor2)

        from fury.window import vtk
        showm = window.ShowManager(scene, size=(1200, 1200))
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        picker.PickTextureDataOn()

        def left_click_callback_mosaic(obj, _ev):
            event_pos = showm.iren.GetEventPosition()

            picker.Pick(event_pos[0],
                        event_pos[1],
                        0,
                        showm.scene)

            # from pdb import set_trace
            # set_trace()
            i, j, k = picker.GetPointIJK()
            print(i, j, k, 'data', data[j, i], 'final', final[j, i], 'final_ma', final_ma[j, i])

        final_actor.AddObserver('LeftButtonPressEvent',
                                left_click_callback_mosaic,
                                1.0)
        data_actor.AddObserver('LeftButtonPressEvent',
                               left_click_callback_mosaic,
                               1.0)

        showm.initialize()
        showm.start()

        # from pylab import imshow, show, figure

        # figure()
        # imshow(data)

        # figure()
        # imshow(final)
        # show()

if __name__ == '__main__':


    from imageio import imread
    from os.path import join as pjoin

    #dname = '/home/jimmy/opto_data/image_frames/average.jpg'
    #fname = pjoin(dname, 'original.jpg')
    dname = '/home/jimmy/opto_data/Image_frames/'
    fname = pjoin(dname, 'average.jpg')

    # fname = pjoin(dname, '188130_web.jpg')
    # fname = pjoin(dname, 'apptop1-10151.jpg')
    # fname = pjoin(dname, 'autumn-landscape-photography-3-900x599.jpg')
    # fname = pjoin(dname, 'forrest.jpg')
    # fname = pjoin(dname, 'Reuben_Wu_Phase_One_XT_1.jpg')
    # fname = pjoin(dname, 'tree_desert.jpg')

    data_orig = imread(fname)
    if len(data_orig.shape) == 3:
        data_orig = np.dot(data_orig[...,:3], [0.2989, 0.5870, 0.1140])
    data_tmp = data_orig
    data_orig = np.zeros(data_orig.shape + (3,))
    data_orig[..., 0] = data_tmp
    data_orig[..., 1] = data_tmp
    data_orig[..., 2] = data_tmp


    # data_orig = np.zeros((512, 512, 3)) + 50

    # I, J, _ = data_orig.shape

    # mI = I // 2
    # mJ = J // 2
    # w = 50
    # for mult in [1, 2, 3, 4, 5]:

    #     data_orig[mI - mult * w: mI + mult * w, mJ - mult * w: mJ + mult * w] += w

    # mult_level = 50
    # noise = mult_level * np.random.rand(*data_orig.shape) - mult_level / 2.
    # data_orig = data_orig + noise

    # images_2d(True, patch_radius=5, bin_size=5, thresholds=[10.])
    # images_2d(True, patch_radius=5, bin_size=5, thresholds=[40.])
    # images_2d(True, patch_radius=3, bin_size=5, thresholds=[15.])
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[1.],
    #           subimage_width=200, histo=True)
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[1.],
    #           subimage_width=None, histo=True)
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[.2],
    #           subimage_width=200, histo=False)
    # images_2d(True, patch_radius=1, bin_size=5, thresholds=[0.5],
    #           subimage_width=None, histo=False,
    #           random_selection=500000,
    #           strides=False)
    # images_2d(True, patch_radius=10, bin_size=5, thresholds=[0.5],
    #           subimage_width=None, histo=False,
    #           random_selection=10000,
    #           strides=True, distance='nssd')

    print('data_orig.max()', data_orig.max())

    # images_2d(data_orig, True, patch_radius=0, bin_size=10, thresholds=[.12],
    #           subimage_width=None, histo=False,
    #           random_selection=None,
    #           strides=False, distance='nsad',
    #           interpolation='min-max', only_centers=False)
    images_2d(data_orig, True, patch_radius=1, bin_size=10, thresholds=[0.02],
              subimage_width=None, histo=False, n_shuffles=50,
              random_selection=None,
              strides=False, distance='nssd',
              interpolation='min-max', only_centers=False)
