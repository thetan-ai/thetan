from time import time
import numpy as np
from thetan.theta import img3d_to_ijkv_features, centroids_from_labels, distance_matrix, img3d_integrals
from dipy.io.image import load_nifti, save_nifti
from dipy.viz.horizon.app import horizon
from dipy.align.reslice import reslice
from dipy.align.scalespace import ScaleSpace
from thetan.berserker import berserker2, theta_g1
from thetan.chaining import chainer, chainer2
from patchify import patchify
from scipy.ndimage import gaussian_gradient_magnitude, convolve, zoom, gaussian_filter
from scipy.ndimage import binary_opening, label, binary_closing, white_tophat, laplace, sobel, binary_dilation, binary_erosion
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
        # median_enh = np.median(data3_enh, axis=-1)

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


# T1 dataset
# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
# fname = '/home/elef/Downloads/T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz'

# fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'

# fname = 'C:/Users/elef/Data/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/S00/t1_ori.nii.gz'
# fname = 'C:/Users/elef/Data/S00/b0_mean.nii.gz'
# fname = 'C:/Users/elef/Data/T1_from_Jasmine/ST18_T1_T1w_MPRAGE_FreeSurfer_0.8iso_20220518080919_8.nii.gz'

# fname = 'C:/Users/elef/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'

# leopard
# fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'
fname = 'C:/Users/Eleftherios/Data/For_Jasmine_Mirdama/T1/T1_T1w_MPRAGE_FreeSurfer_0.8iso_20230126125555_9.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

# DWI dataset
# fname = '/home/elef/.dipy/stanford_hardi/dwi_patch2self.nii.gz'
# data, affine, vxs = load_nifti(fname, return_voxsize=True)
# Just keep b0 for now
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



vxs = np.asarray(vxs)

verbose = True
scale_space = False

if scale_space:
    scs = ScaleSpace(data, 6, affine, vxs)
    data = scs.get_image(2)

data = np.ascontiguousarray(data, dtype='f8')

np.set_printoptions(3, suppress=True)

X = prepare_data(data, affine, perc=(2, 98), mult=300., enh=3., center_dist=False, z_scoring=False, patch=None)

Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

# W10 = W // 100
Wcopy = W.copy()
Xcopy = X.copy()


# Find theta automatically if possible
auto_theta = False
auto_range = (70.0, 150., 10.)
# auto_range = (50.0, 80., 10.)
if auto_theta:
    print('Searching for Theta')
    find_theta(X, *auto_range, W)

gradient_image = True
if gradient_image:
    Wgrad = gaussian_gradient_magnitude(W, sigma=1)
    X[:, 3] += 2 * Wgrad.ravel()

    X2 = X.copy()
    # Removing 4 Wgrads from X[:, 3] provides - 2 * Wgrad of original
    X2[:, 3] -= 4 * Wgrad.ravel()
    Ivol2, Jvol2, Kvol2, W2 = X_to_many_3D(X2, data.shape)

plus_original = True
if plus_original:

    X_orig = X.copy()
    X_orig[:, 3] = Wcopy.ravel()

locality_check = False
if locality_check:
    patch_sz = (3, 3, 3)
    Wpadded = np.pad(Wcopy, patch_sz[0] // 2)
    Wpatch = patchify(Wpadded, patch_sz)
    Wpatch = Wpatch.reshape(data.shape[0] * data.shape[1] * data.shape[2], np.prod(patch_sz))
    Xpatch = np.zeros((Wpatch.shape[0], 4))
    Xpatch[:, 0] = Wpatch.min(axis=-1)
    Xpatch[:, 1] = Wpatch.mean(axis=-1)
    Xpatch[:, 2] = Wpatch.max(axis=-1)
    Xpatch[:, 3] = Wcopy.ravel()
    del Wpatch

    t = 3

# mask_neighb = np.pad(mask, patch[0] // 2)
# mask_neighb = patchify(mask_neighb, patch)

# mask_enh = mask_neighb.reshape((data.shape[0] * data.shape[1] * data.shape[2]),
#                                 (patch[0] * patch[1] * patch[2]))

dog = False
if dog:

    G2 = gaussian_filter(W, sigma=2)
    G1 = gaussian_filter(W, sigma=1)
    G2G1 = G2 - G1

# X[:, 3][X[:, 3] > 900.0] = 900.0

print()

# values used for sims
# theta_value = 12.

# with scale space
# T1 likes 100
# theta_value = 100.
# B0 likes 80?
# theta_value = 80.
# theta_value = 100.

theta_value = 120.
# theta_value = 150.

print('Applying Berserker 2 on X (i,j,k,3w + 2gradW)')
t = time()
data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
print('Duration ', time() - t)
print()

print('Applying Berserker 2 on X (i,j,k,3w - 2gradW)')
t = time()
data_labels2, centroids2 = berserker2_to_3D_labels(X2, theta_value, data_dims=data.shape, verbose=False)
print('Duration ', time() - t)
print()


mask = mask_lt(data_labels, centroids, bk_thr=300)
mask2 = mask_lt(data_labels2, centroids2, bk_thr=300)

if locality_check:

    Xpatch[mask.ravel()==0] = np.zeros(4)
    print('Applying Berserker 2 on Xpatch')
    t = time()
    data_labels_p, centroids_p = berserker2_to_3D_labels(Xpatch, theta_value, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()


print('Centroids\n------')
print_centroids(centroids)

print('Centroids2\n-----')
print_centroids(centroids2)


bk_threshold = False
if bk_threshold:
    print('Labels less than threshold')
    print('---------')
    # bk_idx = np.argwhere(centroids[:, 3] < 360)
    bk_idx = np.argwhere(centroids[:, 3] < 100)
    print(bk_idx)
    print()
    min_bk_idx = bk_idx.min()
    for idx in bk_idx:
        data_labels[data_labels == idx] = min_bk_idx

# dists = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)
# unique_labels, counts = np.unique(data_labels, return_counts=True)

mask3 = patch_sums(mask2, (5, 5, 5))
mask4 = np.zeros(mask3.shape)
# currently very strict - fixed at 0
# mask4[mask3 == 0] = 1
mask4[mask3 < 10] = 1
data_labels_focused, num_focused = label(mask4, structure=np.ones((3, 3, 3)))

print('Focused - Localizing ...')
print('Printing sizes ...')

favorite = analyze_labels(data_labels_focused, num_focused)

mask5 = np.zeros(mask4.shape)
mask5[data_labels_focused == favorite] = 1

# Center of mass calculation for mask 5 (focused)
# This in world coordinates
cx = X[mask5.ravel() == 1, 0].mean()
cy = X[mask5.ravel() == 1, 1].mean()
cz = X[mask5.ravel() == 1, 2].mean()


flood_fill_process = False
if flood_fill_process:
    print('Applying flood fill process ...')

    t = time()

    rand_sz = 100
    fill = 100
    tol = 10

    print('Selecting ', rand_sz, ' seeds')

    # Find ijk coordinates inside the mask
    indices = np.nonzero(mask5)
    indices = np.asarray(indices).T

    # Sample 100
    sample_mask = np.random.randint(0, indices.shape[0], size = rand_sz)
    mask_ff_final = np.zeros(data.shape)
    for i in range(rand_sz):
        flood_seed = tuple(indices[sample_mask[i]])
        mask_ff = flood_fill(Wcopy, flood_seed, fill, tolerance=tol)
        mask_ff_final += mask_ff

    print('Duration ', time() - t)
    print()

distance_field_from_point = False
if distance_field_from_point:
    theta_value_d = 40
    print('Applying Berserker 2 on Xdist')
    t = time()
    Xdist = np.zeros((X.shape[0], 2))
    Xdist[:, 0] = np.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2 + (X[:, 2] - cz) ** 2)
    Xdist[:, 1] = 200 * (1 - mask.ravel())

    data_labels_d, centroids_d = berserker2_to_3D_labels(Xdist, theta_value_d, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

# maskz = zoom(mask, 0.25)
# data_labels_rs = patch_sum_diffs(data_labels_r, (3, 3, 3))


distance_transform = False
if distance_transform:
    from scipy.ndimage import distance_transform_edt, distance_transform_bf, distance_transform_cdt
    dmask_tmp = distance_transform_edt(1 - mask)
    # compare dmask vs 1 - mask

floodfill = False
if floodfill:
    from skimage.morphology import flood_fill

    mask_ff = flood_fill(mask, (115, 76, 160), 50)
    mask_ff2 = flood_fill(mask, (135, 209, 194), 30)
    # TODO
    # two things to check
    # a) use flood or flood_fill on the image gradients (randomly pick 100 seeds)
    # b) use flood fill only on the W[mask==0/1] or similar
    # c) use flood on the localized (focused mask)

watershed = False
if watershed:

    from scipy.ndimage import watershed_ift

    # markers = np.zeros(mask.shape, dtype=np.int16)
    # markers[0, 0, 0] = 1
    # markers[150, 142, 86] = 2
    # markers[107, 106, 75] = 3
    t = time()
    wift = watershed_ift(W.astype('uint16'), mask5.astype('uint8'))

    print('Duration ', time() - t)
    print()

patch_explorations = False
if patch_explorations:

    Wf = patch_features(W, (3, 3, 3))
    Wf_labels, centroids = berserker2_to_3D_labels(Wf, 2000, W.shape, verbose=False)
    Wfsorted = np.sort(Wf, axis=-1)
    Wf_labels2, centroids2 = berserker2_to_3D_labels(Wfsorted, 1000, W.shape, verbose=False)

maskedop = False
if maskedop:
    Xmasked = X.copy()
    Xmasked[:, 3] -= Wgrad.ravel()
    # Xmask[:, 3] = 3 * 300 * (1 - mask.ravel())
    Xmasked[mask.ravel() == 1] = 0

    print('Applying Berserker 2 on the masked X (i, j, k, w)')
    t = time()
    data_labels2, centroids2 = berserker2_to_3D_labels(Xmasked, theta_value * 1.2, data_dims=data.shape, verbose=False)
    print("Duration ", time() - t)

    print('Centroids2')
    print('---------')
    for idx, c in zip(range(centroids2.shape[0]), centroids2[:, 3]):
        print(idx, c)
    print()



from scipy.ndimage import binary_fill_holes, binary_propagation
from scipy.ndimage import binary_opening, binary_closing

# Randomly select voxels
# If region around it is all black then color it.

rand_sel = False
if rand_sel:
    samples = 100
    ijks = np.where(mask.astype(np.bool_) == 0)
    random_selection = np.random.choice(len(ijks[0]), samples, replace=False)
    # mask_grad = gaussian_gradient_magnitude(mask.copy(), sigma=1)
    # labels_grad = gaussian_gradient_magnitude(data_labels_u.copy(), sigma=1)

convolve_ball = False
if convolve_ball:
    print('Multiplying and summing strels - convolution kernel')
    t = time()
    out_strel = np.zeros(mask.shape, dtype='f8')
    strel = ball(10)
    print('Strel shape', strel.shape)
    out_strel = convolve(mask, strel.astype('f8'), mode='constant', cval=0.0)

    # img3d_strel_mult(mask, out_strel, strel.astype('f8'))
    print('Duration ', time() - t)

    # strel = np.ones((41, 41, 41))
    # res = binary_fill_holes(maskb, structure=strel) # origin=(data.shape[0]//2, data.shape[1]//2, data.shape[2]//2))
    # # res = res.astype('f8')
    # print('Starting opening operation using a ball kernel of size', strel.shape)
    # res = binary_closing(maskb, strel)

    # maskb = np.pad(maskb, 20)

    # res, num_features = label(maskb, structure=strel)

    # res = res[21:-20, 21:-20, 21:-20 ]

    # res = flood_fill(maskb, seed_point=(128, 128, 75), new_value=1., selem=strel, connectivity=3)

    mask2 = np.zeros(mask.shape, dtype='f8')
    mask2[out_strel < 300] = 1.0

# print('Ndimage fill holes')
# t = time()
# res = binary_fill_holes(mask, strel)
# print('Duration ', time() - t)

iter_dilation = False
if iter_dilation:
    mask6 = binary_dilation(mask5, iterations=4)
    mask6 = mask6.astype('i4')

difference_check = False
if difference_check:
    mask_diff = mask6 - mask5
    Xd = X2.copy()
    Xd[mask_diff.ravel() < 1] = 0

    print('Applying Berserker 2 on Xd theta/2.')
    t = time()
    data_labels_d, centroids_d = berserker2_to_3D_labels(Xd, theta_value/2., data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids_d)


print('Distance Transform ...')
t = time()
kernel = np.ones((3, 3, 3))
# Note that this is operation on a 1 - mask
maske = binary_erosion(1 - mask, structure=kernel, iterations=1).astype('i4')
mask7 = np.zeros(mask.shape)
edt = distance_transform_edt(maske)
mask7[edt > 2] = 1
print('Duration ', time() - t)
print()

# TODO
# if edt > 2 the coverage is larger and nicer
# but sometimes the mask may connect
# we can do two things to resolve this issue
# a) increase the structure (might be preferrable)
# b) erode the mask
# c) increase edt > 3 or 4

print('Labeling time ...')
t = time()
kernel = np.ones((3, 3, 3))
data_labels_focused2, num_focused2 = label(mask7, structure=kernel)

favorite2 = analyze_labels(data_labels_focused2, num_focused2)
print('Duration ', time() - t)
print()


# invert erosion
print('Inverting erosion ...')
t = time()
final_mask = (data_labels_focused2 == favorite2).astype('i4')
final_mask = binary_dilation(final_mask, structure=kernel, iterations=1).astype('i4')

print('Duration ', time() - t)
print()

verbose = True
if verbose:
    aff = np.eye(4)
    # horizon(images=[(data_labels, aff), (mask, aff), (1 - mask, aff), (Wgrad, aff), (W, aff), (Wcopy, aff)])
    horizon(images=[(final_mask, aff, 'final mask'),
                    (maske, aff, 'maske'),
                    (1 - mask, aff, '1 - mask'),
                    (data_labels_focused2, aff, 'labels_focused2 (edt)'),
                    (edt, aff, 'edt'),
                    (mask7, aff, 'mask7'),
                    (mask5, aff, 'mask5'),
                    (data_labels_focused, aff, 'labels_focused'), (mask4, aff, 'mask4'), (mask3, aff, 'mask3'),
                    (data_labels2, aff, 'labels2'), (mask2, aff, 'mask2'), (data_labels, aff, 'labels'), (mask, aff, 'mask'),
                    (Wgrad, aff, 'Wgrad'), (W, aff, 'W'), (Wcopy, aff, 'Wcopy')])

t = 2
