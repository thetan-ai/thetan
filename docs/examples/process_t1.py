
import numpy as np
from thetan.theta import img3d_to_ijkv_features, centroids_from_labels
from dipy.io.image import load_nifti
from dipy.viz.horizon.app import horizon
from dipy.align.reslice import reslice
from thetan.berserker import berserker2, theta_g1
from thetan.chaining import chainer, chainer2


def find_theta():

    theta_values = []

    sizes = []

    cnt = 0
    for theta_value in np.arange(50, 200, 10):

        print('=========>>', cnt, theta_value)

        centroids, labels = berserker2(X, theta_value, all_out=False)
        theta_values.append(theta_value)
        sizes.append(centroids.shape[0])

        cnt += 1

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(theta_values, sizes)
    plt.show()



# fname = '/home/elef/tmp/IXI002-Guys-0828-T1.nii.gz'
fname = 'C:/Users/Eleftherios/tmp/IXI002-Guys-0828-T1.nii.gz'

data, affine = load_nifti(fname)
value_range = np.percentile(data, q=[2, 99])
data2 = np.interp(data, value_range, (0, 1))
data2 =  300 * np.ascontiguousarray(data2, dtype='f8')

X = img3d_to_ijkv_features(data2)

# X[:, 3] = X[:, 3] ** 2

# Calculating z-scores. It will help with metric spaces

X = (X - np.mean(X, axis=0) / np.std(X, axis=0))

# Intensity more important than location

# X[:, 3] = 4 * X[:, 3]

X[:, 3] = 8 * X[:, 3]

theta_value = 100
# theta_value = 50


# With standard 3*X

# X[:, 3] = 3 * X[:, 3]
# theta_value = 200

# With X[:, 3] ** 2
# X[:, 3] = X[:, 3] ** 2
# theta_value = 32000
# theta_value = 20000

# With X[:, 3] ** 2 / 1000
# theta_value = 100
# theta_value = 80

centroids, labels = berserker2(X, theta_value, all_out=False)

# centroids, labels = theta_g1(X, theta=150, all_out=False)

data_labels = labels.copy()

data_labels = data_labels.reshape(data.shape)

# X3 = X[:, 3].reshape(data.shape)

data_labels = data_labels.astype('f8')

unique_labels, counts = np.unique(data_labels, return_counts=True)

print('>> L', unique_labels[np.argmax(counts)])

# from scipy.ndimage import binary_opening, label


# strel = np.ones((3, 3, 3))

# data_labels[data_labels == unique_labels[np.argmax(counts)]] = 0
# data_labels[data_labels > 0] = 1

# data_labels2 = binary_opening(data_labels, structure=strel)

# data_labels2 = data_labels2.astype('f8')

# data_labels3, num_features = label(data_labels2, structure=None)

# data_labels3 = data_labels3.astype('f8')

# horizon(images=[(data_labels, np.eye(4))])

# horizon(images=[(data_labels, np.eye(4))])

chaining_thr = 1.2 * theta_value

new_labels = chainer(X, centroids, labels, chaining_thr, max_iterations=1)


unique = np.sort(list(set(new_labels)))
new_numbers = range(unique.shape[0])
dix = dict(zip(unique, new_numbers))
new_labels2 = new_labels.copy()

for i, n in enumerate(new_labels):
    new_labels2[i] = dix[new_labels[i]]



data_labels2 = new_labels2.copy()

data_labels2 = data_labels2.reshape(data.shape)

# X3 = X[:, 3].reshape(data.shape)

data_labels2 = data_labels2.astype('f8')

# horizon(images=[(data_labels, np.eye(4))])
# horizon(images=[(data_labels2, np.eye(4))])

# horizon(images=[(data_labels, np.eye(4)), (data_labels2, np.eye(4)), (data2, np.eye(4))])


unique_labels2, counts2 = np.unique(data_labels2, return_counts=True)

new_centroids = centroids_from_labels(X, new_labels2.astype('i8'))

bk = np.where(new_centroids[..., 3] < 500)[0]
fg = np.where(new_centroids[..., 3] >= 500)[0]


binary_mask = data_labels2.copy()

for l in bk:
    binary_mask[data_labels2 == l] = 0
for l in fg:
    binary_mask[data_labels2 == l] = 1

data3 = data2.copy()

data3[binary_mask == 0] = 0


X = img3d_to_ijkv_features(data3)

# X[:, 3] = X[:, 3] ** 2

# Calculating z-scores. It will help with metric spaces

X = (X - np.mean(X, axis=0) / np.std(X, axis=0))

# Intensity more important than location

# X[:, 3] = 4 * X[:, 3]

X[:, 3] = 8 * X[:, 3]

theta_value = 100

centroids, labels = berserker2(X, theta_value, all_out=False)

data_labels = labels.copy()

data_labels = data_labels.reshape(data.shape)

# X3 = X[:, 3].reshape(data.shape)

data_labels = data_labels.astype('f8')

unique_labels, counts = np.unique(data_labels, return_counts=True)

print('>> L', unique_labels[np.argmax(counts)])

chaining_thr = 1.2 * theta_value

new_labels = chainer(X, centroids, labels, chaining_thr, max_iterations=1)

unique = np.sort(list(set(new_labels)))
new_numbers = range(unique.shape[0])
dix = dict(zip(unique, new_numbers))
new_labels2 = new_labels.copy()

for i, n in enumerate(new_labels):
    new_labels2[i] = dix[new_labels[i]]


data_labels2 = new_labels2.copy()

data_labels2 = data_labels2.reshape(data.shape)

# X3 = X[:, 3].reshape(data.shape)

data_labels2 = data_labels2.astype('f8')

# horizon(images=[(data_labels, np.eye(4))])
# horizon(images=[(data_labels2, np.eye(4))])

# horizon(images=[(data_labels, np.eye(4)), (data_labels2, np.eye(4)), (data2, np.eye(4))])


unique_labels2, counts2 = np.unique(data_labels2, return_counts=True)

new_centroids = centroids_from_labels(X, new_labels2.astype('i8'))

morpho = False

if morpho:

    strel = np.ones((7, 7, 7))

    from scipy.ndimage import binary_opening, label
    from skimage.morphology import ball

    strel = ball(3)

    binary_mask2 = binary_opening(binary_mask, structure=strel)
    binary_label2, num_labels = label(binary_mask2) # structure=strel)





t = 1
