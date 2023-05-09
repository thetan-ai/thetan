
import numpy as np
from thetan.theta import img3d_to_ijkv_features
from dipy.io.image import load_nifti
from dipy.viz.horizon.app import horizon
from dipy.align.reslice import reslice
from thetan.berserker import berserker2, theta_g1

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

# With standard X

theta_value = 150 # 8
theta_value = 100 # 24
theta_value = 50 # 52

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
horizon(images=[(data_labels, np.eye(4))])



t = 1
