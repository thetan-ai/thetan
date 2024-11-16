import os
import numpy as np
from glob import glob
from time import time
from thetan.berserker import berserker2_3D_rt, berserker2, prepare_data_3d, X_to_many_3D
from thetan.experimental import clusters_to_3D_labels_custom
from dipy.io.image import load_nifti, save_nifti
# from dipy.viz import horizon
import tracemalloc
import gc
import matplotlib.pyplot as plt
from skimage.segmentation import slic


def aggregate_stats(i, info, seg, data_labels, centroids, rt):

        wm_mask = (seg == 41) | (seg == 2)

        overlap = data_labels[wm_mask == True]

        unq, cnts = np.unique(overlap.ravel(), return_counts=True)

        print('>>>> ', unq, cnts)

        ratio = cnts.max() / wm_mask.sum()
        print('>##> ', ratio)

        info[i, 0] = rt
        info[i, 1] = centroids.shape[0]
        info[i, 2] = ratio
        info[i, 3] = np.prod(data_labels.shape)

        return info


def analyze_results(i, method, data_labels, centroids, centroids_im, rt, info, seg):

    method_dir = method + '/'
    if not os.path.exists(dname + method_dir):
        os.makedirs(dname + method_dir)

    print()
    print('[][][] >> Method used is ', method)
    print()
    aggregate_stats(i, info, seg, data_labels, centroids, rt)
    print('Number of labels ', centroids.shape[0])
    save_nifti(dname + method_dir + subj_id + '_CIM.nii.gz', centroids_im, t1_aff)
    save_nifti(dname + method_dir + subj_id + '_TBL.nii.gz', data_labels, t1_aff)

    print('Runtime ', rt)




def aggregate_results(method, info):
    method_dir = method + '/'
    save_info_fname = dname + method_dir + 'info_' + str(method) + '_' + str(how_many) + '.npy'
    print('Saving result at ', save_info_fname)
    np.save(save_info_fname, info)
    res_array = np.load(save_info_fname)

    x_labels = ['Runtime', 'Clusters', 'Overlap']
    for i, measure in enumerate(x_labels):
        data = [res_array[:, i]]
        fig = plt.figure()
        plt.title(x_labels[i])
        plt.boxplot(data)
        plt.grid()
        fig.savefig(dname + method_dir + x_labels[i] + '_info_' + str(how_many) + '.png', dpi=300)


# dname = '/var/tmp/HCP_labels/'
dname = '/N/slate/elef/HCP_labels/'
search_fname = dname + '*_T1w.nii.gz'

run_experiment = 1
verbose = False
theta_value = 150
K = 5
eps = 20
min_samples = 50

n_segments = 10000
compactness = 0.1

how_many = 5


# save_info_fname = dname + 'info_' + str(method) + '_' + str(how_many) + '.npy'

# print('Summary results will be saved at :')
# print(save_info_fname)

if run_experiment:

    t0 = time()

    res = sorted(glob(search_fname))[:how_many]

    info_tb = np.zeros((how_many, 5))
    info_kmeans = np.zeros((how_many, 5))
    info_bsas = np.zeros((how_many, 5))
    info_dbscan = np.zeros((how_many, 5))
    info_ts = np.zeros((how_many, 5))
    info_slic = np.zeros((how_many, 5))

    for i, fname in enumerate(res):

        subj_id = os.path.basename(fname).split('.nii.gz')[0].split('_')[0]

        print()
        print('>> ' + str(i).zfill(3))
        print('# Participant ID ', subj_id)
        print()

        fname_t1 = dname + subj_id + '_T1w.nii.gz'
        fname_seg = dname + subj_id + '_seg.nii.gz'

        t1, t1_aff = load_nifti(fname_t1)
        seg, seg_aff = load_nifti(fname_seg)

        print('T1 Mbytes ', t1.nbytes // 1024 ** 2)
        print('Seg Kbytes ', seg.nbytes // 1024 )


        data = np.ascontiguousarray(t1, dtype='f8')

        print('Preparing data')
        perc=(0, 98); mult=300.; enh=3.
        affine = t1_aff

        X = prepare_data_3d(data, affine, perc=perc, mult=mult, enh=enh)
        Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)
        save_nifti(dname + subj_id + '_' + '_W.nii.gz', W.astype('f4'), t1_aff)

        data_dims = data.shape

        method = 'berserker2' + '_' + str(theta_value)
        print(f'Starting {method}')
        data_labels, centroids, centroids_im, rt = clusters_to_3D_labels_custom(X, method=method.split('_')[0], theta_value=theta_value, data_dims=data_dims, n_clusters=None, eps=None, min_samples=None, verbose=False, mask=None)
        analyze_results(i, method, data_labels, centroids, centroids_im, rt, info_tb,seg)
        print(f'Finished {method}')

        method = 'kmeans' + '_' + str(K)
        print(f'Starting {method}')
        data_labels, centroids, centroids_im, rt = clusters_to_3D_labels_custom(X, method=method.split('_')[0], theta_value=None, data_dims=data_dims, n_clusters=K, eps=None, min_samples=None, verbose=False, mask=None)
        analyze_results(i, method, data_labels, centroids, centroids_im, rt, info_kmeans, seg)
        print(f'Finished {method}')

        # Memory leaks with large data
        # method = 'bsas' + '_' + str(theta_value) + '_' + str(K)
        # print(f'Starting {method}')
        # data_labels, centroids, centroids_im, rt = clusters_to_3D_labels_custom(X, method=method.split('_')[0], theta_value=theta_value, data_dims=data_dims, n_clusters=K, eps=None, min_samples=None, verbose=False, mask=None)
        # print('Finished clustering one bsas')
        # analyze_results(i, method, data_labels, centroids, centroids_im, rt, info_bsas)
        # print(f'Finished {method}')

        # Memory leaks with large data
        # method = 'dbscan' + '_' + str(eps) + '_' + str(min_samples)
        # print(f'Starting {method}')
        # data_labels, centroids, centroids_im, rt = clusters_to_3D_labels_custom(X, method=method.split('_')[0], theta_value=None, data_dims=data_dims, n_clusters=None, eps=eps, min_samples=min_samples, verbose=False, mask=None)
        # analyze_results(i, method, data_labels, centroids, centroids_im, rt, info_dbscan)
        # print(f'Finished {method}')

        # method = 'ts' + '_' + str(theta_value)
        # print(f'Starting {method}')
        # data_labels, centroids, centroids_im, rt = clusters_to_3D_labels_custom(X, method=method.split('_')[0], theta_value=theta_value, data_dims=data_dims, n_clusters=None, eps=None, min_samples=None, verbose=False, mask=None)
        # analyze_results(i, method, data_labels, centroids, centroids_im, rt, info_ts)
        # print(f'Finished {method}')


        method = 'slic' + '_' + str(n_segments) + '_' + str(compactness)
        print(f'Starting {method}')
        ts_0 = time()
        segments_slic = slic(W, n_segments=n_segments, compactness=compactness, channel_axis=None)
        ts_1 = time()
        u = np.unique(segments_slic)
        centroids_im = np.zeros(W.shape)
        for e in u:
            centroids_im[segments_slic==e] = np.mean(W[segments_slic==e])
        centroids = np.random.rand(n_segments, 3)
        data_labels = segments_slic.astype('f4')
        rt_slic = ts_1 - ts_0
        analyze_results(i, method, data_labels, centroids, centroids_im, rt_slic, info_slic, seg)
        print(f'Finished {method}')


    print('Total duration ', time() - t0)

    method = 'berserker2' + '_' + str(theta_value)
    aggregate_results(method, info_tb)

    method = 'kmeans' + '_' + str(K)
    aggregate_results(method, info_kmeans)

    # method = 'bsas' + '_' + str(theta_value) + '_' + str(K)
    # aggregate_results(method, info_bsas)

    # method = 'dbscan' + '_' + str(eps) + '_' + str(min_samples)
    # aggregate_results(method, info_dbscan)

    # method = 'ts' + '_' + str(theta_value)
    # aggregate_results(method, info_ts)

    method = 'slic' + '_' + str(n_segments) + '_' + str(compactness)
    aggregate_results(method, info_slic)

