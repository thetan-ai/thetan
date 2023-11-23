
import numpy as np
from thetan.theta import img3d_to_ijkv_features, centroids_from_labels
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


def find_theta(X, start, stop, step):

    theta_values = []
    sizes = []

    cnt = 0
    print('Looking for nice theta values')
    for theta_value in np.arange(start, stop, step):

        print('=========>>', cnt, theta_value)
        centroids, labels = berserker2(X, theta_value, all_out=False)
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

# T1 dataset
fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
# fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'
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

gradient_image = False
if gradient_image:
    data = gaussian_gradient_magnitude(data, sigma=1)

verbose = True

# from dipy.align.reslice import reslice
# data2, affine2 = reslice(data, affine, vxs, vxs * 3.)
# data3, affine3 = reslice(data2, affine2, vxs * 3, vxs)

scale_space = False
if scale_space:
    scs = ScaleSpace(data, 6, affine, vxs)
    data = scs.get_image(2)

data = np.ascontiguousarray(data, dtype='f8')



np.set_printoptions(3, suppress=True)

# TODO check carefully the range of values in X
# Then also try with simpler data such as homocentric spheres
# X = prepare_data(data, affine, perc=False, mult=300., enh=3., center_dist=False, z_scoring=False)

# Recommendation for T1
# X = prepare_data(data, affine, perc=(2, 99), mult=300., enh=3., center_dist=False, z_scoring=False)

# Recommendation for B0
# X = prepare_data(data, affine, perc=(2, 99), mult=300., enh=1., center_dist=False, z_scoring=False)

# Recommendation for Binf or RL
# X = prepare_data(data, affine, perc=(2, 98), mult=300., enh=3., center_dist=False, z_scoring=False, patch=(3, 3, 3))

X = prepare_data(data, affine, perc=(2, 98), mult=300., enh=3., center_dist=False, z_scoring=False, patch=None)


print('Looking at the range of values per feature')
print('Feature, min, max, mean, std')
tmp_array = np.zeros((X.shape[1], 5))
for i in range(X.shape[1]):
    tmp_array[i] = np.array([i, X[:, i].min(), X[:, i].max(), X[:, i].mean(), X[:, i].std()])

print(tmp_array)


# Find theta automatically if possible
auto_theta = 0
auto_range = (70.0, 150., 10.)
if auto_theta:
    find_theta(X, *auto_range)

print()
print('Starting b2ch1 #1')

# values used for sims
# theta_value = 12.
# chaining_factor = 1.6

# theta_value = 25.
# chaining_factor = 1.4

# values used for real data

# scale space
# theta_value = 200.
# chaining_factor = 1.2

# with scale space
# T1 likes 100
# theta_value = 100.
# B0 likes 80?
# theta_value = 80.
# gradient image
theta_value = 120.

chaining_factor = 1.2
bk_thr = 250
# Binf liked 100
bk_thr2 = 100
# RL liked ?
bk_thr = 225
bk_thr2 = 25

# assume only intensity is important
# X[:, 0] = X[:, 3]
# X[:, 1] = X[:, 3]
# X[:, 2] = X[:, 3]

binary_mask, data_labels, new_centroids, data_labels_orig, centroids_orig, X, Y, Z, W = b2ch1(X, theta_value=theta_value, chaining_factor=chaining_factor, bk_thr=bk_thr, sh=data.shape)


print('Original Berserker Centroids')
print(centroids_orig)

print('With label id')
print(np.concatenate((np.arange(centroids_orig.shape[0])[:, None], centroids_orig), axis=-1))

print()
print('New centroids')
print(new_centroids)

# Remove low intensities
tmp = data_labels.copy()
indx = np.where(new_centroids[:, 3] < bk_thr2)[0]
for indx_ in indx:
    tmp[tmp == indx_] = 0

# data3 = data2.copy()
# data3[binary_mask == 0] = 0

# print()
# print('Starting b2ch1 #2')
# binary_mask2, data_labels2, new_centroids2, data_labels_orig2, centroids_orig2 = b2ch1(data3, affine, theta_value=100, chaining_factor = 1.2, bk_thr=250)

# What I did that seemed promissing is ...
# data_labels2[data_labels2 == 0] = 0
# data_labels2[data_labels2 == 1] = 0
# this may indicate that binary_mask/2 may not be truly useful

# simple division in 10 parts can be useful (0, 900)
Wd = np.floor(W / 100.)
Wd[Wd == 0] = 0
Wd[Wd == 1] = 0
Wd[Wd == 2] = 0
Wd[Wd == 3] = 0

if verbose:
    # horizon(images=[(data_labels_orig, np.eye(4)), (X, np.eye(4)), (Y, np.eye(4)), (Z, np.eye(4)), (W, np.eye(4)), (data, np.eye(4))])
    # horizon(images=[(data_labels, np.eye(4)), (data_labels_orig, np.eye(4)), (X, np.eye(4)), (Y, np.eye(4)), (Z, np.eye(4)), (W, np.eye(4)), (data, np.eye(4))])

    aff = np.eye(4)
    # horizon(images=[((tmp > 0).astype('f4'), aff), (tmp, aff), (data_labels, aff), (data_labels_orig, aff), (Wd, aff), (W, aff), (data, aff)])
    horizon(images=[(data_labels_orig, np.eye(4)), (W, np.eye(4))])

morpho = False

if morpho:

    # opening followed by closing can remove unneeded objects
    # strel = np.ones((7, 7, 7))

    binary_mask = tmp > 0



    strel = ball(9)

    Wl = laplace(W)
    Wg = gaussian_gradient_magnitude(W, sigma=1)
    # Ws = sobel(W)
    # I need to study tophat better
    # Wt9 = white_tophat(W, structure=strel)

    binary_mask2 = binary_opening(binary_mask, structure=strel)
    binary_mask3 = binary_closing(binary_mask2, structure=strel)


    # binary_label2, num_labels = label(binary_mask2) # structure=strel)
    if verbose:
        horizon(images=[(tmp, np.eye(4)), (binary_mask2.astype('i4'), np.eye(4)), (binary_mask3.astype('i4'), np.eye(4)), (W, np.eye(4)), (data, np.eye(4))])
    save_nifti(fmask, binary_mask3.astype('i4'), affine)

t = 1
