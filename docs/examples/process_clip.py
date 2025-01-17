import numpy as np
from time import time
from thetan.berserker import berserker2_clean as tb
from thetan.theta import assign, distance_matrix

from sklearn import metrics
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from munkres import Munkres
from pylab import figure, plot, show, legend



dpath = '/home/elef/Data/CLIP/'

fitrain = dpath + 'CIFAR-10_image_embedding_train.npy'

fitest = dpath + 'CIFAR-10_image_embedding_test.npy'

fconcat = dpath + 'concat_embedding.npy'

fgt_train = dpath + 'CIFAR-10_labels_train.txt'
fgt_test = dpath + 'CIFAR-10_labels_test.txt'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import pairwise_distances


def distillation(X, n_clusters=10):
    # Simulated Data
    # X = np.random.rand(500, 10)  # 500 samples, 10 features

    # Teacher Model (K-Means)
    teacher = KMeans(n_clusters=n_clusters, random_state=42)
    tf = teacher.fit(X)
    soft_labels = teacher.transform(X)  # Soft assignments (distance to centroids)

    # Student Model (Neural Network)
    student = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
    student.fit(X, np.argmax(-soft_labels, axis=1))  # Mimic teacher's clusters

    # Predictions
    student_clusters = student.predict(X)
    print("Student Clusters:", np.unique(student_clusters))

    return student_clusters


def normalization():
    pass

def johnson_lindenstrauss_reduction(X):

    # Generate high-dimensional data
    # np.random.seed(42)
    n_points = X.shape[0]  # Number of points
    original_dim = X.shape[1]  # Original dimension
    # X = np.random.rand(n_points, original_dim)  # Random high-dimensional dataset

    # Target dimension (lower-dimensional space)
    epsilon = 0.1  # Distortion factor
    target_dim = int(np.ceil(4 * np.log(n_points) / (epsilon**2)))

    # Random projection matrix
    R = np.random.randn(target_dim, original_dim) / np.sqrt(target_dim)

    # Perform the random projection
    X_reduced = np.dot(X, R.T)

    return X_reduced


def johnson_lindenstrauss_expansion(X, high_dim):

    # Generate high-dimensional data
    # np.random.seed(42)
    n_points = X.shape[0]  # Number of points
    low_dim = X.shape[1]  # Original dimension

    # Target high dimension for feature expansion
    high_dim = high_dim  # Expanded dimension

    # Random projection matrix for expansion
    R = np.random.randn(high_dim, low_dim) / np.sqrt(high_dim)

    # Perform the random projection (feature expansion)
    X_expanded = np.dot(X, R.T)

    return X_expanded


def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    print("[Clustering Result]: NMI = {:.2f}, ARI = {:.2f}".format(nmi * 100, ari * 100))
    return nmi, ari


def feature_triuk(feature, metric='euclidean'):

    matrix = distance_matrix(feature, feature, metric=metric)
    feature_space = matrix[np.triu_indices(matrix.shape[0], k = 1)]
    return feature_space


def keep_largest(X, centroids, labels, nicest=10):
    uniq, sizes = np.unique(labels, return_counts=True)
    niceK = nicest
    biggest_k = np.argsort(sizes)[-niceK:]
    hd_centroids = np.ascontiguousarray(centroids[biggest_k].copy())
    labels2 = assign(X, hd_centroids)
    return labels2, sizes


def distance_matrix_approach(theta=0.44, metric='arccos', skip=10, verbose=False):

    X = np.load(fitest).astype('f8')
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    gt_labels = np.loadtxt(fgt_test)

    centroids, labels = tb(X, theta, metric=metric)

    evaluate(gt_labels, labels)
    print('#Clusters', centroids.shape)

    labels2, sizes = keep_largest(X, centroids, labels, 10)
    evaluate(gt_labels, labels2)

    uniq2, sizes2 = np.unique(labels2, return_counts=True)

    labels3, sizes = keep_largest(X, centroids, labels, 18)
    evaluate(gt_labels, labels3)

    uniq3, sizes3 = np.unique(labels3, return_counts=True)

    if verbose:


        figure()
        plot(sorted(sizes)[::-1], label='sizes')
        plot(sorted(sizes2)[::-1], label='sizes2')
        plot(sorted(sizes3)[::-1], label='sizes3')
        legend()
        show()


def standard_approach():

    X = np.load(fitest).astype('f8')

    gt_labels = np.loadtxt(fgt_test)

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    skip = 10

    # labels = distillation(X, 10)
    # evaluate(gt_labels, labels)

    res0 = feature_triuk(X[0][:, None])[::skip]
    print(res0.shape)

    res1 = feature_triuk(X[1][:, None])[::skip]
    print(res1.shape)

    t = time()
    X2 = np.zeros((X.shape[0], res0.shape[0]))
    for i, x in enumerate(X):
        X2[i] = feature_triuk(x[:, None])[::skip]
    print('Feature Expansion Duration ', time() - t)

    X2 = np.ascontiguousarray(X2)

    theta = 9.8

    # for theta in range(9, 10, 12):
    metric = 'euclidean'

    theta = 200
    metric = 'l1_norm'

    theta = 0.4
    metric = 'arccos'


    print('Theta ', theta, ' Metric ', metric)

    t = time()
    centroids, labels = tb(X2, theta, metric=metric)
    print('TB Duration ', time() - t)

    evaluate(gt_labels, labels)

    print('#Clusters', centroids.shape)

    labels2, sizes = keep_largest(X, centroids, labels, 10)
    evaluate(gt_labels, labels2)


def jl_approach(theta=0.44, metric='arccos', edim=2000, verbose=False, normalize=False):

    print()
    print('Starting Johnson Lindenstrauss Approach .. ')
    print('Theta ', theta, ' Metric ', metric, ' Expanded Dimension ', edim)

    X = np.load(fitest).astype('f8')

    print('X min', X.min(), 'X max', X.max())

    if normalize:
        # X = X / np.linalg.norm(X, axis=1, keepdims=True)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X2 = johnson_lindenstrauss_expansion(X, edim)

    print('X', X.shape, 'X2', X2.shape)

    gt_labels = np.loadtxt(fgt_test)

    centroids, labels = tb(X2, theta, metric=metric)

    evaluate(gt_labels, labels)
    print('#Clusters', centroids.shape)

    labels2, sizes = keep_largest(X2, centroids, labels, 10)
    evaluate(gt_labels, labels2)
    print('#Clusters', labels2.max() + 1)

    uniq2, sizes2 = np.unique(labels2, return_counts=True)

    labels3, sizes = keep_largest(X2, centroids, labels, 20)
    evaluate(gt_labels, labels3)
    print('#Clusters', labels3.max() + 1)

    uniq3, sizes3 = np.unique(labels3, return_counts=True)

    if verbose:


        figure()
        plot(sorted(sizes)[::-1], label='sizes')
        plot(sorted(sizes2)[::-1], label='sizes2')
        plot(sorted(sizes3)[::-1], label='sizes3')
        legend()
        show()



if __name__ == '__main__':

    # standard_approach()

    # jl_approach(0.44, 'arccos', 2000, False)

    # jl_approach(0.44, 'arccos', 4000, False)

    # jl_approach(0.44, 'arccos', 6000, False)


    jl_approach(10.0, 'l1_norm', 1024, False, True)

    jl_approach(8.0, 'l1_norm', 1024, False, True)

    jl_approach(12.0, 'l1_norm', 1024, False, True)

    t = 1