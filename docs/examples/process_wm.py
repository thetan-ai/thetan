import os
import numpy as np
from glob import glob
from time import time
from thetan.berserker import berserker2_3D_rt, berserker2
from dipy.io.image import load_nifti, save_nifti
# from dipy.viz import horizon
import tracemalloc
import gc
import matplotlib.pyplot as plt

# X = np.random.rand(10000000, 4)
# print('Mbytes', X.nbytes // 1024 ** 2)

# cnt = 0
# theta_value = 0.2
# while (True):
#     print('>>>', cnt)
#     t = time()
#     centroids, labels = berserker2(X, theta_value, metric='euclidean', all_out=False, verbose=False)
#     cnt += 1
#     print('Duration ', time() - t, ' ', centroids.shape[0])



dname = '/var/tmp/HCP_labels/'
search_fname = dname + '*_T1w.nii.gz'

run_experiment = True
verbose = False
theta_value = 150  # 120
save_results = False
how_many = 10 # 1000
analyze_results = False

save_info_fname = dname + 'info_' + str(theta_value) + '_' + str(how_many) + '.npy'

if run_experiment:

    t = time()

    res = sorted(glob(search_fname))[:how_many]

    info = np.zeros((how_many, 5))

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

        # tracemalloc.start()
        # amem_, peakmem_ = tracemalloc.get_traced_memory()
        # amem_ = amem_ // (1024 ** 2)
        data_labels, centroids, centroids_im, W, rt = berserker2_3D_rt(t1, t1_aff, theta_value=theta_value, perc=(0, 98), mult=300., enh=3.)
        # tmp = berserker2_3D_rt(t1, t1_aff, theta_value=theta_value, perc=(0, 98), mult=300., enh=3.)
        # amem, peakmem = tracemalloc.get_traced_memory()
        # peakmem = peakmem // (1024 ** 2)
        # amem = amem // (1024 ** 2)
        # tracemalloc.stop()

        # print('>>>> Peakmem before ', peakmem_)
        # print('>>>> Peakmem after ', peakmem)
        # print('>>>> Acurrent before ', amem_)
        # print('>>>> Acurrent after ', amem)

        print('Number of labels ', centroids.shape[0])

        # if verbose:
        #     images = [(t1, t1_aff, 't1'), (seg, seg_aff, 'seg'), (data_labels, t1_aff, 'dl'), (centroids_im, t1_aff, 'cim'), (W, t1_aff, 'W')]
        #     horizon(images=images)

        wm_mask = (seg == 41) | (seg == 2)

        overlap = data_labels[wm_mask == True]

        unq, cnts = np.unique(overlap.ravel(), return_counts=True)

        print('>>>> ', unq, cnts)

        ratio = cnts.max() / wm_mask.sum()
        print('>##> ', ratio)

        info[i, 0] = rt
        info[i, 1] = centroids.shape[0]
        info[i, 2] = ratio
        info[i, 3] = np.prod(t1.shape)

        # if ratio > 0.90:

        #     # from dipy.viz import horizon
        #     # aff = t1_aff
        #     # images = [(t1, aff, 't1'), (W, aff, 't1'), (centroids_im, aff, 'centroids_im'), (seg, aff, 'seg')]
        #     # horizon(images=images)

        t = 1
        t = 2

        print('Saving images ...')
        print(dname + subj_id + '_W.nii.gz')
        print(W.nbytes, W.dtype, W.shape)
        print(centroids_im.nbytes, centroids_im.dtype, centroids_im.shape)
        print(data_labels.nbytes, data_labels.dtype, data_labels.shape)

        save_nifti(dname + subj_id + '_W.nii.gz', W, t1_aff)
        save_nifti(dname + subj_id + '_CIM.nii.gz', centroids_im, t1_aff)
        save_nifti(dname + subj_id + '_TBL.nii.gz', data_labels, t1_aff)

    if save_results:
        print('Saving result at ', save_info_fname)
        np.save(save_info_fname, info)

    print('Total duration ', time() - t)


if analyze_results:

    res_array = np.load(save_info_fname)

    x_labels = ['Runtime', 'Clusters', 'Overlap']
    # x = np.arange(len(x_labels))

    for i, measure in enumerate(x_labels):
        data = [res_array[:, i]]
        fig = plt.figure()
        plt.title(x_labels[i])
        plt.boxplot(data)
        # plt.xticks(x, x_labels)
        plt.grid()

        # plt.show()
        fig.savefig(dname + x_labels[i] +  '.png', dpi=300)






t = 1
t = 2