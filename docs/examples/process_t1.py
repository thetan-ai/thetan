
import numpy as np
from thetan.theta import img3d_to_ijkv_features, centroids_from_labels
from dipy.io.image import load_nifti
from dipy.viz.horizon.app import horizon
from dipy.align.reslice import reslice
from thetan.berserker import berserker2, theta_g1
from thetan.chaining import chainer, chainer2


def prepare_data(data, affine, mult=300, enh=4):

    value_range = np.percentile(data, q=[2, 99])
    data2 = np.interp(data, value_range, (0, 1))
    data2 =  mult * np.ascontiguousarray(data2, dtype='f8')

    X = img3d_to_ijkv_features(data2)

    # move to world coordinates W (xyz)

    W = np.dot(affine[:3, :3], X[:, :3].T) + affine[:3, 3][:, None]
    X[:, :3] = W.T

    # add distance from world coordinates origin

    Wnorm = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2)
    Xenh = np.concatenate((X, Wnorm[:, None]), axis=-1)
    X = Xenh
  
    # Calculating z-scores. It may help with metric spaces
    X = (X - np.mean(X, axis=0) / np.std(X, axis=0))

    # Calculating pseudo z-scores. It may help with metric spaces.
    # Assuming mean at 0 for all columns

    # X = (X / np.std(X, axis=0))

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
    plt.grid()
    plt.show()


def b2ch1(X, theta_value=100, chaining_factor = 1.2, bk_thr=250):


    # Deploying berserker v2

    centroids, labels = berserker2(X, theta_value, all_out=False)

    data_labels = labels.copy()

    data_labels = data_labels.reshape(data.shape)

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

    return binary_mask, data_labels2, new_centroids, data_labels, centroids


# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'

data, affine, vxs = load_nifti(fname, return_voxsize=True)

vxs = np.asarray(vxs)

# from dipy.align.reslice import reslice

# data2, affine2 = reslice(data, affine, vxs, vxs * 3.)
# data3, affine3 = reslice(data2, affine2, vxs * 3, vxs)

scale_space = True

if scale_space:

    from dipy.align.scalespace import ScaleSpace

    scs = ScaleSpace(data, 6, affine, vxs)
    data = scs.get_image(2)



print()
print('Starting b2ch1 #1')

# TODO check carefully the range of values in X
# Then also try with simpler data such as homocentric spheres
X = prepare_data(data, affine, mult=300, enh=4)

for i in range(5):
    print(">>>>", i)
    print(X[:, i].min())
    print(X[:, i].max())


# Find theta automatically if possible

find_theta(X, 50.0, 150.0, 25.0)

binary_mask, data_labels, new_centroids, data_labels_orig, centroids_orig = b2ch1(X, theta_value=125., chaining_factor = 1.2, bk_thr=250)

print('Original Berserker Centroids')
np.set_printoptions(3, suppress=True)
print(centroids_orig)

# data3 = data2.copy()
# data3[binary_mask == 0] = 0

# print()
# print('Starting b2ch1 #2')
# binary_mask2, data_labels2, new_centroids2, data_labels_orig2, centroids_orig2 = b2ch1(data3, affine, theta_value=100, chaining_factor = 1.2, bk_thr=250)

# What I did that seemed promissing is ...
# data_labels2[data_labels2 == 0] = 0
# data_labels2[data_labels2 == 1] = 0
# this may indicate that binary_mask/2 may not be truly useful

# data_labels2[binary_mask == 0] = 
horizon(images=[(data_labels_orig, np.eye(4)), (data, np.eye(4))])
# horizon(images=[(data_labels, np.eye(4)), (data_labels2, np.eye(4)), (data_labels_orig, np.eye(4)), (data2, np.eye(4))])


morpho = False

if morpho:

    strel = np.ones((7, 7, 7))

    from scipy.ndimage import binary_opening, label
    from skimage.morphology import ball

    strel = ball(3)

    binary_mask2 = binary_opening(binary_mask, structure=strel)
    binary_label2, num_labels = label(binary_mask2) # structure=strel)



t = 1
