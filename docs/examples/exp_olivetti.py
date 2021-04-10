import logging
from time import time
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as pl
from thetan.theta import theta_g1, assign
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
from sklearn.cluster import KMeans
from theta_machines_tests import theta_g1, theta_g2, theta_g2x, labels_to_centroids
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction.image import extract_patches_2d

def show_image(faces, target, patches_size):
    for f in range(faces.shape[0]):
        face0 = faces[f].reshape((patches_size, patches_size))
        target0 = target[f]
        pl.figure()
        pl.title(target0)
        pl.imshow(face0)
        pl.show()

# #############################################################################
# Load faces data
faces, target = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                     random_state=RandomState(0))

# for f in range(faces.shape[0]):
#     face0 = faces[f].reshape((64, 64))
#     target0 = target[f]
#     pl.figure()
#     pl.title(target0)
#     pl.imshow(face0)
#     pl.show()
faces = faces.astype(np.float64)
face0 = faces[0].reshape((64, 64))
target0 = target[0]
faces_orig = faces.copy()
n_samples, n_features = faces.shape
# faces = faces - faces.mean(axis=0)
test = 1

#######################K-means++##########################
from theta_machines_tests import theta_g1, theta_g2, theta_g2x
methods = [] # ['kmeans', 'kmeans++', 'fgn_ma', 'birch', 'mb_kmeans', 'gmm']
if 'kmeans++' in methods:
    timings = {}
    evaluation_thr = 0.5
    np.random.seed(0)
    t1 = time()
    kmeans = KMeans(n_clusters=40, init='k-means++', n_init=10,
                    max_iter=300, tol=0.0001, random_state=0, n_jobs=1).fit(faces_orig)
    t2 = time()
    kmeans_time = t2 - t1
    print('kmeans++ ', kmeans_time)
    kcentroids = kmeans.cluster_centers_
    klabels = kmeans.labels_
    timings['kmeans++'] = kmeans_time
    tmp = 0
    test_fig = kcentroids.reshape(40,64,64)

t = 1
########################################################
samples_total = 400
patches_size = 25
# all_small_c = np.zeros((samples_total, 291 * patches_size*patches_size), dtype='f8')
all_small_c = []
list_centroids = []
verbose = False
random_patches = 100
total_random_patches = samples_total * random_patches

patch_sz = (patches_size, patches_size)

all_patches = np.zeros((total_random_patches, patches_size*patches_size))

for i in range(samples_total):

    patches = extract_patches_2d(faces[i].reshape(64, 64), patch_sz, max_patches=random_patches, random_state=42)
    patches = patches.reshape(random_patches, patches_size*patches_size)
    all_patches[i*random_patches: i*random_patches + random_patches] = patches

# centroids, labels = theta_g2(all_patches, theta=.6, num_shuffles=5)
t1 = time()
centroids, labels = theta_g1(all_patches, theta=4.3)
unique_labels, sizes = np.unique(labels, return_counts=True)
t2 = time()
print('Duration for one run', t2 - t1)

verbose = False

if verbose:
    num_clusters = []
    # range_theta = np.arange(0.1, 5, 0.1)
    range_theta = np.arange(4, 5, 0.1)
    cnt = 0
    for theta_t in range_theta:
        print(cnt)
        centroids, labels = theta_g1(all_patches, theta=theta_t) #labels.shape:(3903,)
        num_clusters.append(len(centroids))
        cnt += 1


    pl.figure()
    num_clusters = np.asarray(num_clusters)
    pl.plot(range_theta, num_clusters)
    pl.plot(range_theta, num_clusters, 'o')
    pl.show()


t = 1

for i in range(samples_total):

    patches = extract_patches_2d(faces[i].reshape(64, 64), patch_sz, max_patches=0.1, random_state=42)
    patches = patches.reshape(291, patches_size*patches_size)
    centroids, labels = theta_g2(patches, theta=.6, num_shuffles=5)

    # if verbose:
    #     num_clusters = []
    #     range_theta = np.arange(0.1, 5, 0.2)
    #     for theta_t in range_theta:
    #         centroids, labels = theta_g1(patches, theta=theta_t)
    #         num_clusters.append(len(centroids))

    #     pl.figure()
    #     num_clusters = np.asarray(num_clusters)
    #     pl.plot(range_theta, num_clusters)
    #     pl.grid()
    #     pl.show()
    #     print(centroids.shape[0])
    #     t = 1
    unique_labels, sizes = np.unique(labels, return_counts=True)
    print('Centoids length', (len(centroids)))

    small_idx = np.argsort(sizes)[::-1][:10]
    small_centroids = centroids[small_idx] #small_centroids.shape : (24,121)



    all_small_c.append(small_centroids.reshape(1, 10 *  patches_size ** 2)) #all_small_c.shape : (3267,)

all_small_c = np.concatenate(all_small_c)
show_image(centroids, labels, patches_size = 11)
###########Second Round g2################
centroids, labels = theta_g2(all_small_c, theta=4.566, num_shuffles=5)

# show_clusters_gt(faces_orig, 'gt labels',klabels, kcentroids)
# def show_clusters_gt(X, method, labels_k, kcentroids)
verbose = False
if verbose:
    num_clusters = []
    range_theta = np.arange(0.1, 5, 0.1)
    for theta_t in range_theta:
        centroids, labels = theta_g1(all_small_c, theta=theta_t) #labels.shape:(3903,)
        num_clusters.append(len(centroids))

    pl.figure()
    num_clusters = np.asarray(num_clusters)
    # pl.plot(range_theta, num_clusters)
    # pl.plot(range_theta, num_clusters, 'o')
    pl.show()
    t = 1

test_point = 1


# for i in range(10):
#     pl.figure()
#     pl.imshow(patches[i])
# pl.show()

