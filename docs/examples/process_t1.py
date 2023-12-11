
import numpy as np
from thetan.theta import img3d_to_ijkv_features, centroids_from_labels, distance_matrix
from dipy.io.image import load_nifti, save_nifti
from dipy.viz.horizon.app import horizon
from dipy.align.reslice import reslice
from dipy.align.scalespace import ScaleSpace
from thetan.berserker import berserker2, theta_g1
from thetan.chaining import chainer, chainer2
from patchify import patchify
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.ndimage import binary_opening, label, binary_closing, white_tophat, laplace, sobel, gaussian_gradient_magnitude


def expand_voxels(data, patch_size=(3, 3, 3)):

    return patchify(data, patch_size)

def prepare_data(data, affine, perc=(2, 99), mult=300, enh=2, center_dist=False, z_scoring=True, patch=None):

    if perc is None:
        data2 = data
    else:
        value_range = np.percentile(data, q=[2, 99])
        data2 = np.interp(data, value_range, (0, 1))
        data2 =  mult * np.ascontiguousarray(data2, dtype='f8')

    # This simply adds median information
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


# T1 dataset
# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'

# fname = 'C:/Users/elef/Data/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/elef/Data/S00/t1_ori.nii.gz'
# fname = 'C:/Users/elef/Data/S00/b0_mean.nii.gz'
# fname = 'C:/Users/elef/Data/T1_from_Jasmine/ST18_T1_T1w_MPRAGE_FreeSurfer_0.8iso_20220518080919_8.nii.gz'

fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/Data/IXI-T1/IXI012-HH-1211-T1.nii.gz'


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

W10 = W // 100
Wcopy = W.copy()


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

X[:, 3] += Wgrad.ravel()

print()
print('Starting Thetan Berserker2 #1')

# values used for sims
# theta_value = 12.

# with scale space
# T1 likes 100
# theta_value = 100.
# B0 likes 80?
# theta_value = 80.
# gradient image
# theta_value = 100.
theta_value = 150.

data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)

## data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)

print('Centroids')
print('---------')
for idx, c in zip(range(centroids.shape[0]), centroids[:, 3]):
    print(idx, c)
print()

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

if verbose:
    aff = np.eye(4)    
    # horizon(images=[(Wgrad, aff), ((Wgrad + W, aff)) (Wgrad // 20, aff), (data_labels, aff), (W, aff), (W10, aff)])
    horizon(images=[(data_labels, aff), (W, aff), (Wcopy, aff)])

dists = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)

unique_labels, counts = np.unique(data_labels, return_counts=True)

# Remove low intensities
bk_thr = 250
tmp = data_labels.copy()
indx = np.where(centroids[:, 3] < bk_thr)[0]
min_indx = np.argmin(centroids[:, 3])

for indx_ in indx:
    tmp[tmp == indx_] = min_indx

X[:, 3][tmp.ravel() == min_indx] = 0

mask = tmp == indx_

data_labels2, centroids2 = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)

if verbose:
    aff = np.eye(4)    
    horizon(images=[(data_labels2, aff), (data_labels, aff), (W, aff), (Wcopy, aff)])

# Select random points in the label to see distance from the origin


# Select random patches from labels

patch = (3, 3, 3)
data_3x3x3 = np.pad(W.reshape(data.shape), patch[0]//2)
data_enh = patchify(data_3x3x3, patch)

unique_labels2, counts2 = np.unique(data_labels2, return_counts=True)

samples = 100

mins = []
maxs = []

for l in unique_labels2.astype('i8'):

    ijks = np.where(data_labels == l)

    random_selection = np.random.choice(len(ijks[0]), samples, replace=False)
    i_s = ijks[0][random_selection]
    j_s = ijks[1][random_selection]
    k_s = ijks[2][random_selection]
    ijks_rs = np.vstack([i_s, j_s, k_s]).T

    print('Current label >> ', l)
    for rec in range(samples):
        # print(data_enh[tuple(ijks_rs[rec])].ravel())
        min_ = np.floor(data_enh[tuple(ijks_rs[rec])].min())
        max_ = np.floor(data_enh[tuple(ijks_rs[rec])].max())
        mins.append(min_)
        maxs.append(max_)


    print()

    t = 1


import matplotlib.pyplot as plt

plt.figure()
plt.plot(mins, maxs, '*')
plt.grid()
plt.show()

t = 2