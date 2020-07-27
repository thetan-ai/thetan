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
    #elif distance.lower() == 'nsid':
    #    metric = NormalizedSumInfDiffMetric(feature=IdentityFeature())
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


    no_clusters = len(clusters)
    no_clusters_ma = len(clusters_ma)
    cluster_colors = distinguishable_colormap(nb_colors=no_clusters)
    cluster_colors_ma = distinguishable_colormap(nb_colors=no_clusters_ma)

    if patch_radius > 0:
        print('Removed extra padding')
        data = data[patch_radius: -patch_radius, patch_radius: -patch_radius]

    final = np.zeros(data.shape)
    final_ma = np.zeros(data.shape)

    for ind in range(no_clusters):
        final[ind//data.shape[1], ind%data.shape[1]] = cluster_colors[clusters[ind]]

    for ind in range(no_clusters_ma):
        final_ma[ind//data.shape[1], ind%data.shape[1]] = cluster_colors_ma[clusters_ma[ind]]


    return final, final_ma, data, clusters, clusters_ma


#@profile
def images_2d(data_orig, interactive=True, patch_radius=1, bin_size=5,
              thresholds=[2.], n_shuffles=10,
              subimage_width=None, lab_color=False, histo=True,
              random_selection=1000000, strides=False, distance='nsad',
              interpolation='98p', only_centers=True):

    final, final_ma, data, clusters, clusters_ma = fgn_image_2d_rgb(data_orig, patch_radius=patch_radius,
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
        return clusters, clusters_ma

if __name__ == '__main__':


    from imageio import imread
    from os.path import join as pjoin

    # directory name and file name
    dname = r'C:\Users\Jong Sung Park\Desktop\collab_king\01'
    fname = pjoin(dname, 'average.jpg')

    data_orig = imread(fname)
    # change to gray scale if image is color
    # the rates of values follow scipy
    if len(data_orig.shape) == 3:
        data_orig = np.dot(data_orig[...,:3], [0.2989, 0.5870, 0.1140])
    data_tmp = data_orig
    data_orig = np.zeros(data_orig.shape + (3,))
    data_orig[..., 0] = data_tmp
    data_orig[..., 1] = data_tmp
    data_orig[..., 2] = data_tmp

    print('data_orig.max()', data_orig.max())

    # added retun values for saving data for later use
    clusters, clusters_ma = images_2d(data_orig, True, patch_radius=1, bin_size=10, thresholds=[0.01],
              subimage_width=None, histo=False, n_shuffles=50,
              random_selection=None,
              strides=False, distance='nssd',
              interpolation='min-max', only_centers=False)

    # saving data for later use
    """
    cluster_file = open("cluster", "w")
    cluster_ma_file = open("cluster_ma", "w")
    for idx in clusters:
        cluster_file.write(str(idx) + " ")
    for idx in clusters_ma:
        cluster_ma_file.write(str(idx) + " ")

    cluster_file.close()
    cluster_ma_file.close()
    """