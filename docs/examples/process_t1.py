from time import time
import numpy as np
from thetan.theta import img3d_to_ijkv_features, centroids_from_labels, distance_matrix, img3d_integrals
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import horizon
from dipy.align.reslice import reslice
from dipy.align.scalespace import ScaleSpace
from thetan.berserker import berserker2, theta_g1, berserker2_brain_seg
from thetan.chaining import chainer, chainer2
from patchify import patchify
from scipy.ndimage import gaussian_gradient_magnitude, convolve, zoom, gaussian_filter
from scipy.ndimage import binary_opening, label, binary_closing, white_tophat, laplace, sobel, binary_dilation, binary_erosion, morphological_gradient
from skimage.segmentation import flood, flood_fill
from skimage.morphology import ball
from scipy.ndimage import distance_transform_edt, distance_transform_bf, distance_transform_cdt


def expand_voxels(data, patch_size=(3, 3, 3)):

    return patchify(data, patch_size)


def patch_sum_diffs(data, patch):
    """ Sum differences in linearized patches
    """

    data2 = np.pad(data, patch[0]//2)
    data2_enh = patchify(data2, patch)
    data2_enh = data2_enh.reshape(data2_enh.shape[0],
                                  data2_enh.shape[1],
                                  data2_enh.shape[2],
                                  patch[0] * patch[1] * patch[2])

    return np.sum(np.diff(data2_enh, axis=-1), axis=-1)

def patch_sums(data, patch):
    """ Sums in linearized patches
    """

    data2 = np.pad(data, patch[0]//2)
    data2_enh = patchify(data2, patch)
    data2_enh = data2_enh.reshape(data2_enh.shape[0],
                                  data2_enh.shape[1],
                                  data2_enh.shape[2],
                                  patch[0] * patch[1] * patch[2])

    return np.sum(data2_enh, axis=-1)


def patch_features(data, patch, step):
    """ Create a feature vector
    """
    data2 = np.pad(data, patch[0]//2)
    data2_enh = patchify(data2, patch, step)
    data2_enh = data2_enh.reshape(np.prod(data2_enh.shape[:3]), patch[0] * patch[1] * patch[2])

    return data2_enh

def patch_down(data, patch, step):
    """ Downsample using steps
    """
    data_step = patchify(data, patch, step=step)
    data_step_shape = data_step.shape
    data_step2 = data_step.reshape(data_step_shape[0],
                                   data_step_shape[1],
                                   data_step_shape[2],
                                   np.prod(data_step_shape[3:]))
    data_step2 = data_step2.mean(axis=-1)

    return data_step2

def prepare_data(data, affine, perc=(2, 99), mult=300, enh=2, center_dist=False, z_scoring=True, patch=None):

    if perc is None:
        data2 = data
    else:
        value_range = np.percentile(data, q=[2, 99])
        data2 = np.interp(data, value_range, (0, 1))
        data2 =  mult * np.ascontiguousarray(data2, dtype='f8')

    # This simply adds median information
    # Currently not being used
    if patch is not None:
        data3 = np.pad(data2, patch[0]//2)
        data3_enh = patchify(data3, patch)
        data3_enh = data3_enh.reshape((data2.shape[0] * data2.shape[1] * data2.shape[2]),
                                      (patch[0] * patch[1] * patch[2]))
        mean_enh = np.mean(data3_enh, axis=-1)
        median_enh = np.median(data3_enh, axis=-1)

    X = img3d_to_ijkv_features(data2)

    if patch is not None:
        X[:, 3] = median_enh

    # move to world coordinates W (xyz)

    W = np.dot(affine[:3, :3], X[:, :3].T) + affine[:3, 3][:, None]
    X[:, :3] = W.T

    # add distance from world coordinates origin

    if center_dist:

        Wnorm = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)
        Xenh = np.concatenate((X, Wnorm[:, None]), axis=-1)
        X = Xenh

    if z_scoring:
        # Calculating z-scores. It may help with metric spaces
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Intensity more important than location
    X[:, 3] = enh * X[:, 3]

    return X


def find_theta(X, start, stop, step, data, matrix_calc=False):

    theta_values = []
    sizes = []
    current_L2_sums = []

    cnt = 0
    print('Looking for nice theta values')
    for theta_value in np.arange(start, stop, step):

        print('=========>>', cnt, theta_value)
        centroids, labels = berserker2(X, theta_value, all_out=False)

        data_labels = labels.copy()
        data_labels = data_labels.reshape(data.shape)
        data_labels = data_labels.astype('f8')

        # horizon(images=[(data_labels, np.eye(4)), (data, np.eye(4))])

        if matrix_calc:

            D = distance_matrix(centroids, centroids, metric='euclidean')
            Dsort = np.sort(D, axis=-1)
            current_L2_sum = np.sum(Dsort[:, 1])
            # current_L2_sum = np.sum(np.min(D, axis=-1))
            print('Current L2 sum C matrix is ', current_L2_sum)
            current_L2_sums.append(current_L2_sum)

        theta_values.append(theta_value)
        sizes.append(centroids.shape[0])
        cnt += 1

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(theta_values, sizes, '*-')
    plt.xlabel('theta values')
    plt.ylabel('sizes')
    plt.grid()
    plt.show()

    if matrix_calc:
        plt.figure()
        plt.plot(theta_values, current_L2_sums, '*-')
        plt.xlabel('theta values')
        plt.ylabel('current L2 sum of min with centroids')
        plt.grid()
        plt.show()

    t = 1

def volumetric_sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0


def simulate_volumes(sh=(100, 100, 100), radii=[10, 20], values=[10, 20], pos=[(50, 50, 50), (50, 50, 50)]):

    data = np.zeros(sh)

    for rad_, vals_, pos_ in zip(radii, values, pos):

        tmp = volumetric_sphere(sh, radius=rad_, position=pos_)
        data[tmp] += vals_

    data += np.random.rand(*data.shape)

    # np.array([[1, 0, 0, 50]])
    return data, np.eye(4), (1., 1., 1.)


def b2ch1(X, theta_value=100, chaining_factor = 1.2, bk_thr=250, sh=None):


    # Deploying berserker v2

    centroids, labels = berserker2(X, theta_value, all_out=False)

    data_labels = labels.copy()

    data_labels = data_labels.reshape(sh)

    data_labels = data_labels.astype('f8')

    # horizon(images=[(data_labels, np.eye(4)), (data, np.eye(4))])

    unique_labels, counts = np.unique(data_labels, return_counts=True)

    print('>> L', unique_labels[np.argmax(counts)])

    chaining_thr = chaining_factor * theta_value

    # Deploying chainer (only 1st iteration)

    new_labels = chainer(X, centroids, labels, chaining_thr, max_iterations=1)

    unique = np.sort(list(set(new_labels)))
    new_numbers = range(unique.shape[0])
    dix = dict(zip(unique, new_numbers))
    new_labels2 = new_labels.copy()

    for i, n in enumerate(new_labels):
        new_labels2[i] = dix[new_labels[i]]

    data_labels2 = new_labels2.copy()
    data_labels2 = data_labels2.reshape(data.shape)
    data_labels2 = data_labels2.astype('f8')

    unique_labels2, counts2 = np.unique(data_labels2, return_counts=True)
    new_centroids = centroids_from_labels(X, new_labels2.astype('i8'))

    print('Focus on intensities for this level')
    print(new_centroids[..., 3])

    bk = np.where(new_centroids[..., 3] < bk_thr)[0]
    fg = np.where(new_centroids[..., 3] >= bk_thr)[0]

    binary_mask = data_labels2.copy()
    print(bk)
    print(fg)

    for l in bk:
        binary_mask[data_labels2 == l] = 0
    for l in fg:
        binary_mask[data_labels2 == l] = 1

    return binary_mask, data_labels2, new_centroids, data_labels, centroids, X[:, 0].reshape(sh), X[:, 1].reshape(sh), X[:, 2].reshape(sh), X[:, 3].reshape(sh)


def berserker2_to_3D_labels(X, theta_value, data_dims, verbose=False):
    # Random storm

    # Deploying berserker v2

    centroids, labels = berserker2(X, theta_value, all_out=False)

    data_labels = labels.copy()

    data_labels = data_labels.reshape(data_dims)

    data_labels = data_labels.astype('f8')

    # horizon(images=[(data_labels, np.eye(4)), (data, np.eye(4))])

    # unique_labels, counts = np.unique(data_labels, return_counts=True)

    # print('>> L', unique_labels[np.argmax(counts)])

    return data_labels, centroids


def X_to_many_3D(X, data_dims):

    X0 = X[:, 0].reshape(data_dims)
    X1 = X[:, 1].reshape(data_dims)
    X2 = X[:, 2].reshape(data_dims)
    X3 = X[:, 3].reshape(data_dims)

    return X0, X1, X2, X3


def print_ranges(X):

    print('Looking at the range of values per feature')
    print('Feature, min, max, mean, std')
    tmp_array = np.zeros((X.shape[1], 5))
    for i in range(X.shape[1]):
        tmp_array[i] = np.array([i, X[:, i].min(), X[:, i].max(), X[:, i].mean(), X[:, i].std()])
    print(tmp_array)


def looking_for_big_objects(data_labels, labeled_array, unique_labels):

    big_objects = {}

    for idx in unique_labels:

        print('Label index ==> ', idx, ' largest object')

        # Find largest objects
        labeled_array, num_features = label(data_labels == int(idx))
        print(num_features)

        unique_labels_la, counts_la = np.unique(labeled_array, return_counts=True)
        print(counts_la)

        # this is incorrect (argmax not enough) and ignoring background
        tmp_array = labeled_array == counts_la[1:].argmax() + 1
        big_objects[idx] = tmp_array.copy()
        print(np.sum(big_objects[idx]))

        aff = np.eye(4)
        horizon(images=[(big_objects[idx].astype('i4'), aff), (data_labels, aff), (W, aff)])

        t = 2

    print()


def mask_lt(data_labels, centroids, bk_thr=300):
    """ Remove low intensities using label centroids
    """
    # bk_thr = 300 # 400 or 250 interesting too
    data_labels_u = data_labels.copy()
    indx = np.where(centroids[:, 3] < bk_thr)[0]
    min_indx = np.argmin(centroids[:, 3])

    for indx_ in indx:
        data_labels_u[data_labels == indx_] = min_indx

    # X[:, 3][tmp.ravel() == min_indx] = 0
    mask = data_labels_u == min_indx
    maskb = mask.copy()
    mask = mask.astype('f8')
    return mask

def mask_gt(data_labels, centroids, bk_thr=300):
    """ Keep high intensities using label centroids
    """
    # bk_thr = 300 # 400 or 250 interesting too
    data_labels_u = data_labels.copy()
    indx = np.where(centroids[:, 3] > bk_thr)[0]
    max_indx = np.argmax(centroids[:, 3])

    for indx_ in indx:
        data_labels_u[data_labels == indx_] = max_indx

    mask = data_labels_u == max_indx
    maskb = mask.copy()
    mask = mask.astype('f8')
    return mask


def print_centroids(centroids):
    for idx, c in zip(range(centroids.shape[0]), centroids[:, 3]):
        print(idx, c)
    print()


def analyze_labels(data_labels_focused, num_focused):

    sizes = []
    for i in range(num_focused):
        cur_sz = (data_labels_focused == i).sum()
        sizes.append(cur_sz)

    sizes = np.asarray(sizes)
    print(sizes)
    favorites = np.argsort(sizes)[::-1][:3]
    favorite = favorites[1] # 0 most likely bk
    print('Prefered localized index')
    print(favorite, sizes[favorite])

    return favorite


def watershed_wgrad(Wgrad):

    watersh = False
    markers = 1000
    compactness = 0.01
    print('Watershed on grad mag')
    t = time()
    from skimage.segmentation import watershed
    Wshed = watershed(Wgrad, markers=markers, compactness=compactness)
    # TODO
    # try also with Sobel if available in 3D
    print('Duration ', time() - t)
    print()


# T1 dataset
# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
# fname = '/home/elef/Downloads/T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz'
# fname = '/home/elef/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'

# fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'

# fname = 'C:/Users/elef/Data/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/S00/t1_ori.nii.gz'
# fname = 'C:/Users/elef/Data/S00/b0_mean.nii.gz'
# fname = 'C:/Users/elef/Data/T1_from_Jasmine/ST18_T1_T1w_MPRAGE_FreeSurfer_0.8iso_20220518080919_8.nii.gz'

# fname = 'C:/Users/elef/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'

# leopard
# fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/For_Jasmine_Mirdama/T1/T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz'
# fname = 'C:/Users/Eleftherios/.dipy/stanford_hardi/t1.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

# DWI dataset
# fname = '/home/elef/.dipy/stanford_hardi/dwi_patch2self.nii.gz'
# fname = 'C:/Users/Eleftherios/.dipy/stanford_hardi/HARDI150.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# # Just keep b0 for now
# data = data[..., 0].copy()

# Binfinity
# fname = '/home/elef/Data/distortions/b0_u.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/distortions/b0_u.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# fmask = '/home/elef/Data/distortions/b0_u_thetan2.nii.gz'

# B0 HCP (LR)
# fname = '/home/elef/Data/distortions/103818_3T_DWI_dir96_LR.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/distortions/103818_3T_DWI_dir96_LR.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# fmask = '/home/elef/Data/distortions/103818_3T_DWI_dir96_LR_thetan2.nii.gz'
# fmask = 'C:/Users/Eleftherios/Data/distortions/103818_3T_DWI_dir96_LR_thetan2.nii.gz'
# data = data[..., 0].copy()

# Simulation example
# data, affine, vxs = simulate_volumes(sh=(100, 100, 100), radii=[10, 20, 30, 40], values=[10, 20, 30, 40.], pos=[(50, 50, 50), (50, 50, 50), (50, 50, 50), (50, 50, 50)])
# data, affine, vxs = simulate_volumes(sh=(100, 100, 100), radii=[10, 20, 30, 40], values=[5, 5, 5, 5.], pos=[(50, 50, 50), (50, 50, 50), (50, 50, 50), (50, 50, 50)])

print('Starting Thetan Berserker 2 for Brain Segmentation ...')
t = time()
images = berserker2_brain_seg(data, affine, return_all=True)
print('Duration ', time() - t)
print()

print('Automated quality assurance')


horizon(images=images)


t = 2

print('Done')

