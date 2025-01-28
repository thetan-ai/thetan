import numpy as np
from time import time
from thetan.berserker import berserker2_clean as tb
from thetan.berserker import tbscan_half, tb_of_tb
from thetan.theta import assign, distance_matrix

from sklearn import metrics
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from munkres import Munkres
from pylab import figure, plot, show, legend
from sklearn.decomposition import PCA


dpath = '/home/elef/Data/CLIP/'
fitrain = dpath + 'CIFAR-10_image_embedding_train.npy'
fitest = dpath + 'CIFAR-10_image_embedding_test.npy'
fconcat = dpath + 'concat_embedding.npy'

fgt_train = dpath + 'CIFAR-10_labels_train.txt'
fgt_test = dpath + 'CIFAR-10_labels_test.txt'

dpath2 = '/home/elef/Data/ImageNet1K/'
fitest2 = dpath2 + 'ImageNet_image_embedding_test.npy'
fgt_test2 = dpath2 + 'ILSVRC2012_validation_ground_truth.txt'


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

def johnson_lindenstrauss_reduction(X, low_dim):

    # Generate high-dimensional data
    # np.random.seed(42)
    n_points = X.shape[0]  # Number of points
    original_dim = X.shape[1]  # Original dimension
    # X = np.random.rand(n_points, original_dim)  # Random high-dimensional dataset

    # Target dimension (lower-dimensional space)
    epsilon = 0.1  # Distortion factor
    # target_dim = int(np.ceil(4 * np.log(n_points) / (epsilon**2)))
    target_dim = low_dim  # Lower dimension

    # Random projection matrix
    R = np.random.randn(target_dim, original_dim) / np.sqrt(target_dim)

    # Perform the random projection
    X_reduced = np.dot(X, R.T)

    t = 1

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


def evaluate(label, pred, num_clusters):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    print("[Clustering Result]: NMI = {:.2f}, ARI = {:.2f} NUM = {:d}".format(nmi * 100, ari * 100, num_clusters))
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





def jl_approach(theta_value=0.44, metric='arccos', edim=2000, verbose=False, normalize=False, expand=True):

    print()
    print('Starting Johnson Lindenstrauss Approach .. ')
    print('Theta ', theta_value, ' Metric ', metric, ' Target Dimension ', edim)

    X = np.load(fitest).astype('f8')

    if normalize:
        # X = X / np.linalg.norm(X, axis=1, keepdims=True)
        scaler = StandardScaler(with_mean=True, with_std=False)
        X = scaler.fit_transform(X)
        X = 1000 * X # let's escape from potential numerical issues

    if expand:
        X2 = johnson_lindenstrauss_expansion(X, edim)
    else:
        X2 = johnson_lindenstrauss_reduction(X, edim)

    print('X', X.shape, 'X2', X2.shape)
    print('X min', X.min(), 'X max', X.max(), 'X2 min', X2.min(), 'X2 max', X2.max())

    gt_labels = np.loadtxt(fgt_test)

    centroids, labels = tb(X2, theta_value, metric=metric)

    evaluate(gt_labels, labels)
    print('#Clusters ', centroids.shape)

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


def jl_experiments(main_theta=0.44):

    for a in np.linspace(1.0, 1.5, 10):

        print()

        print('>>>> Factor' , a)

        theta_value = main_theta * a

        jl_approach(theta_value, 'arccos', 256, normalize=True, expand=False) # X/2

        jl_approach(theta_value, 'arccos', 128, normalize=True, expand=False) # X/4

        jl_approach(theta_value, 'arccos', 64, normalize=True, expand=False) # X/8

        jl_approach(theta_value, 'arccos', 1024, normalize=True, expand=True) # 2X

        jl_approach(theta_value, 'arccos', 2048, normalize=True, expand=True) # 4X

        jl_approach(theta_value, 'arccos', 3072, normalize=True, expand=True) # 6X


def standard_approach(theta_value, metric, thresh=20, normalize=True, sparsify=True, triuk=False, jl=True):

    X = np.load(fitest).astype('f8')

    gt_labels = np.loadtxt(fgt_test)

    if normalize:
        # X = X / np.linalg.norm(X, axis=1, keepdims=True)
        scaler = StandardScaler(with_mean=True, with_std=False)
        X = scaler.fit_transform(X)
        X = 1000 * X # let's escape from small numbers


    if sparsify:
        X[X < thresh] = 0

    if jl:
        R = np.random.randn(*X.shape)
        X = X * R


    X = np.ascontiguousarray(X)

    print('X shape ', X.shape, ' X min ', X.min(), ' X max ', X.max(), ' X mean ', X.mean(), ' X std ', X.std())

    # labels = distillation(X, 10)
    # evaluate(gt_labels, labels)
    if triuk:

        skip = 10

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
        X = X2

    print('Theta ', theta_value, ' Metric ', metric)

    t = time()
    centroids, labels = tb(X, theta_value, metric=metric)
    print('TB Duration ', time() - t)

    num_clusters = centroids.shape[0]
    evaluate(gt_labels, labels, num_clusters)

    # print('# Clusters ', centroids.shape)

    labels2, sizes = keep_largest(X, centroids, labels, 10)
    evaluate(gt_labels, labels2, 10)


def analysis_approach(fX, fgt, theta_value, metric, normalize=True, verbose=False, ndim=None, keep=10, epsilon=0.001):

    X = np.load(fX).astype('f8')

    gt_labels = np.loadtxt(fgt)

    if ndim is not None:

        pca = PCA(n_components=ndim)
        X = pca.fit_transform(X)

    X = np.ascontiguousarray(X)



    # X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # outer_sums = np.zeros(X.shape[0])
    # for i, x in enumerate(X):
    #     outer_sums[i] = np.sum(np.outer(x, x)**2)

    # if normalize:
    #     # X = X / np.linalg.norm(X, axis=1, keepdims=True)
    #     scaler = StandardScaler(with_mean=True, with_std=False)
    #     X = scaler.fit_transform(X)
    #     X = 1000 * X # let's escape from small numbers

    # m = X.mean(axis=0)
    # X2 = np.ascontiguousarray(100000 * (X - m))

    Xnorms = np.linalg.norm(X, axis=1)

    print('X shape ', X.shape, ' X min ', X.min(), ' X max ', X.max(), ' X mean ', X.mean(), ' X std ', X.std())
    print('Theta ', theta_value, ' Metric ', metric)

    # if verbose:
    #     figure()
    #     plot(sorted(Xnorms)[::-1])

    print('>>> TB ')

    t = time()
    centroids, labels = tb(X, theta_value, metric=metric)
    print('TB Duration ', time() - t)

    # print('#Clusters ', centroids.shape[0])

    # t = time()
    # centroids3, labels3 = tb_of_tb(centroids, labels, 1.2 * theta_value, metric=metric)
    # print('TB of TB Duration ', time() - t)

    # print('#Clusters ', centroids3.shape[0])

    print('>>> TBSCAN ')
    best_eps = -1
    best_min_samples = -1
    maxc = -1
    maxc_ari = -2

    max_a = -2
    best_eps_a = -1
    best_min_samples_a = -1
    best_clusters_a = -1


    for eps in np.linspace(3.25, 3.28, 400):

        for min_samples in np.arange(2, 10, 1):

            t = time()
            centroids2, labels2 = tbscan_half(X, eps=eps, min_samples=min_samples, initial_centroids=centroids, initial_labels=labels)
            print('Eps', eps, ' min_samples', min_samples, ' #clusters ', centroids2.shape[0], ' Duration ', time() - t)

            t = time()
            nmi, ari = evaluate(gt_labels, labels2, centroids2.shape[0])
            print('Evaluation Duration ', time() - t)


            if centroids2.shape[0] > maxc:
                maxc = centroids2.shape[0]
                best_eps = eps
                best_min_samples = min_samples
                maxc_ari = ari

            if ari > max_a:
                max_a = ari
                best_eps_a = eps
                best_min_samples_a = min_samples
                best_clusters_a = centroids2.shape[0]


    print('>>> Summary \n\n')

    print('Best Eps', best_eps, ' Best min_samples', best_min_samples, ' Max Clusters ', maxc, ' Max ARI', maxc_ari)

    print('Best Eps A', best_eps_a, ' Best min_samples A ', best_min_samples_a, ' Max Clusters A ', best_clusters_a, ' Max ARI', max_a)

    return

    print('Centroid Neighbors')
    no_cneighbors = np.zeros(centroids.shape[0])
    for i, c in enumerate(centroids):
        c2c = distance_matrix(c[None, :], centroids, metric=metric).ravel()
        # no_cneighbors[i] = np.sum((c2c < 0.5 * theta_value) & (c2c > epsilon))
        no_cneighbors[i] = np.sum((c2c <  theta_value) & (c2c > 0.95 * theta_value))

    print('Medians')
    medians_min = np.zeros(centroids.shape[0])
    medians_max = np.zeros(centroids.shape[0])


    # Save distance values
    for i, c in enumerate(centroids):
        X_tmp = np.ascontiguousarray(X[labels == i])
        c2X = distance_matrix(c[None, :], X_tmp, metric=metric).ravel()
        medians_min[i] = np.argmin(c2X)
        medians_max[i] = np.argmax(c2X)

    num_clusters = centroids.shape[0]
    evaluate(gt_labels, labels, num_clusters)

    # TODO: TB of TB, TBSCAN



    labels2, sizes = keep_largest(X, centroids, labels, keep)
    evaluate(gt_labels, labels2, keep)

    return centroids, labels, labels2, no_cneighbors, medians_min, medians_max


    # if verbose:
    #     show()

    t = 1

if __name__ == '__main__':


    fX = fitest2
    fgt = fgt_test2

    analysis_approach(fX, fgt, 3.77, 'euclidean', normalize=True, verbose=True, ndim=50, keep=1000)
    # centroids, labels, labels2, no_cneighbors, medians_min, medians_max = res


    # from pylab import plot, show, title, figure, hist, subplot, imshow, cm, colorbar, legend, xlim, ylim, scatter
    # figure(); title('cneib'); plot(no_cneighbors); figure(); title('medians min'); plot(medians_min); figure(); title('medians max'); plot(medians_max); show();

    t = 1

    if False:
        theta_value = 4.
        metric = 'euclidean'
        dim = 20

        # for theta_value in np.linspace(3.6, 3.8, 20):

        for theta_value in 10 * [3.75,]:

            print()
            print('>>>> Theta Value', theta_value)
            print()
            res = analysis_approach(theta_value, metric, normalize=True, verbose=True, ndim=dim)
            centroids, labels, labels2, no_cneighbors, medians_min, medians_max = res

            print()


        theta_value = 2.
        metric = 'euclidean'
        dim = 15

        for theta_value in np.linspace(2.4, 4., 100):

            # for theta_value in 10 * [3.75,]:

            print()
            print('>>>> Theta Value', theta_value)
            print()
            res = analysis_approach(theta_value, metric, normalize=True, verbose=True, ndim=dim)
            centroids, labels, labels2, no_cneighbors, medians_min, medians_max = res

            print()


    t = 1
